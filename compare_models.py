# -*- coding: utf-8 -*-
"""
compare_models.py — 같은 문제로 여러 모델을 채점해 나란히 놓습니다

둘째 날 학습 전후를 비교하려고 만든 그 채점표를, 이번에는 **모델끼리** 비교하는 데 씁니다.
느낌이 아니라 숫자로 고르기 위해서입니다.

  py -3.12 compare_models.py --models llama3.1:8b ornith-1.5:9b
  py -3.12 compare_models.py --models a b --file eval.jsonl --save 결과.json
  py -3.12 compare_models.py --models a b --show-fails      틀린 답을 그대로 보여줍니다

공정하게 재기 위해 지킨 것
  · 시스템 프롬프트는 llm_arm_bridge에서 그대로 가져옵니다 (여기서 다시 쓰지 않습니다)
  · temperature 0 — 채점기가 매번 다른 점수를 내면 비교가 무의미합니다
  · 대화 기억 없음 — 문제끼리 서로 힌트를 주지 않게, 각 문제를 독립으로 냅니다
  · 실행 가능성 판정은 bridge.validate() 하나로 합니다 (규칙을 두 곳에 두지 않습니다)

평가 파일(eval.jsonl)의 한 줄
  {"instruction": "집게를 세 번 딱딱거리시오", "output": {"commands": [...]}}
"""

import argparse
import json
import os
import re
import sys
import time

import requests

import llm_arm_bridge as bridge

EVAL_FILE = "eval.jsonl"

CRITERIA = [
    ("valid",     "실행 가능한 형태인가",   "bridge.validate()를 통과하는가"),
    ("serial",    "팔의 동작이 같은가",     "★ 실제로 아두이노에 나가는 줄이 기대와 같은가 — 표현이 달라도 됨"),
    ("exact",     "글자까지 완전 일치",     "JSON 표현까지 똑같은가 (참고용. 낮아도 문제 아님)"),
    ("direction", "앞/뒤 규약",             "기대 답의 delta 부호와 같은 방향인가"),
    ("relative",  "상대 이동 사용",         "기대가 move를 쓰면 실제도 move를 쓰는가"),
    ("sequence",  "단발로 때우지 않는가",   "몸짓 지시에 명령을 3개 이상 냈는가"),
    ("count",     "횟수",                   "지시문의 'n번'과 repeat times가 맞는가"),
]

NUM_KO = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6,
          "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
GESTURE = ("인사", "숙여", "흔들", "도리도리", "춤", "시늉", "제스처", "몸짓", "웨이브")


# ── 평가 문제 읽기 ──────────────────────────────────────────────────────

def load_problems(path):
    """eval.jsonl 을 읽습니다. 열쇠 이름이 조금 달라도 받아들입니다."""
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠ {path} {n}번째 줄을 읽지 못했습니다: {e}")
                continue
            instr = row.get("instruction") or row.get("input") or row.get("지시")
            out = row.get("output") or row.get("expected") or row.get("정답")
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except json.JSONDecodeError:
                    out = None
            cmds = None
            if isinstance(out, dict):
                cmds = out.get("commands")
            elif isinstance(out, list):
                cmds = out
            if instr:
                items.append({"instruction": instr, "expected": cmds})
    return items


# ── 모델에게 묻기 ───────────────────────────────────────────────────────

def ask(model, instruction, timeout=300, budget=1024, think=False):
    """한 문제를 냅니다. (명령리스트 | None, 걸린시간, 원문 또는 오류)

    ★ think=False 가 중요합니다.
      Qwen 3.x 계열(Ornith 포함)은 답하기 전에 추론 토큰을 뱉습니다. 그대로 두면
      토큰 예산을 생각하는 데 다 쓰고 JSON이 잘리거나 아예 비어서 나옵니다.
      추론은 message.content 가 아니라 message.thinking 에 담기므로,
      content만 읽던 예전 판은 "JSON이 아님: (빈 문자열)"만 보게 됩니다.
      — 모델이 못한 게 아니라 측정기가 잘못 물어본 것이었습니다.
    """
    t0 = time.time()
    body = {
        "model": model,
        "messages": [{"role": "system", "content": bridge.SYSTEM_PROMPT},
                     {"role": "user", "content": instruction}],
        "stream": False,
        "format": "json",
        # ★ temperature 0 — 브리지의 기본값(0.2)과 다릅니다. 비교에는 재현성이 우선입니다.
        "options": {"temperature": 0, "num_predict": budget,
                    "repeat_penalty": 1.0, "num_ctx": 8192},
    }
    if not think:
        body["think"] = False
    try:
        resp = requests.post(bridge.OLLAMA_URL, json=body, timeout=timeout)
    except requests.ConnectionError:
        return None, time.time() - t0, "Ollama에 연결하지 못했습니다 (ollama serve 확인)"
    dt = time.time() - t0

    if resp.status_code == 404:
        return None, dt, f"모델 '{model}' 없음 — ollama pull {model}"
    if resp.status_code == 400 and "think" in resp.text.lower():
        # 추론 수준을 문자열로만 받는 모델이 있습니다 (예: gpt-oss)
        body["think"] = "low"
        try:
            resp = requests.post(bridge.OLLAMA_URL, json=body, timeout=timeout)
        except requests.ConnectionError:
            return None, dt, "Ollama에 연결하지 못했습니다"
    if resp.status_code != 200:
        return None, dt, f"HTTP {resp.status_code}: {resp.text[:140]}"

    data = resp.json()
    msg = data.get("message", {})
    raw = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()
    cut = (data.get("done_reason") == "length")

    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()

    if not raw:
        if thinking:
            return None, dt, (f"답 없이 추론만 냈습니다 ({len(thinking)}자). "
                              f"--budget 을 올리거나 --think 를 확인하세요")
        return None, dt, ("출력이 비었습니다"
                          + (" — 토큰 예산 소진(--budget 을 올리세요)" if cut else ""))
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        if cut:
            return None, dt, f"토큰 예산 소진으로 잘림 (--budget 을 올리세요): {raw[:90]}"
        return None, dt, f"JSON이 아님: {raw[:120]}"

    cmds = obj.get("commands") if isinstance(obj, dict) else obj
    if not isinstance(cmds, list):
        return None, dt, f"commands 배열이 없음: {raw[:120]}"
    return cmds, dt, raw


# ── 채점 ────────────────────────────────────────────────────────────────

def norm(cmds):
    """비교를 위해 정규화합니다. 순서는 뜻이 있으므로 유지합니다."""
    return json.dumps(cmds, ensure_ascii=False, sort_keys=True)


def deltas(cmds):
    out = []
    for c in cmds or []:
        if isinstance(c, dict):
            if c.get("cmd") == "move" and "delta" in c:
                out.append(c["delta"])
            out.extend(deltas(c.get("commands")))
    return out


def uses_move(cmds):
    for c in cmds or []:
        if isinstance(c, dict):
            if c.get("cmd") == "move":
                return True
            if uses_move(c.get("commands")):
                return True
    return False


def repeat_times(cmds):
    for c in cmds or []:
        if isinstance(c, dict):
            if c.get("cmd") == "repeat" and "times" in c:
                return c["times"]
            got = repeat_times(c.get("commands"))
            if got is not None:
                return got
    return None


def wanted_count(instruction):
    """지시문에서 '3번' 또는 '세 번'을 읽습니다. 없으면 None.

    ★ "팔을 1번 꺾으시오" 는 반복이 아니라 한 동작입니다. 이걸 repeat 로 세면
      멀쩡한 답을 틀렸다고 셉니다(실제로 그랬습니다). 그래서 2회 이상이거나
      '반복'이라는 말이 있을 때만 횟수 기준을 적용합니다.
    """
    n = None
    m = re.search(r"(\d+)\s*번", instruction)
    if m:
        n = int(m.group(1))
    else:
        for word, k in NUM_KO.items():
            if re.search(word + r"\s*번", instruction):
                n = k
                break
    if n is None:
        return None
    if n >= 2 or "반복" in instruction:
        return n
    return None


def grade(problem, got):
    """한 문제의 채점 결과. 해당 없는 항목은 None (평균에서 제외)."""
    instr, exp = problem["instruction"], problem["expected"]
    r = {k: None for k, _n, _d in CRITERIA}

    try:
        bridge.validate(json.loads(json.dumps(got)))
        r["valid"] = True
    except Exception:
        r["valid"] = False
        return r                       # 실행조차 못 하면 나머지는 볼 것도 없습니다

    if exp is not None:
        r["exact"] = (norm(got) == norm(exp))
        # ★ 실제로 팔에 나가는 줄로 비교합니다. wait ms:5000 과 wait seconds:5 는
        #   JSON은 다르지만 아두이노에 가는 것은 같습니다. 팔이 하는 일이 같으면 맞은 것입니다.
        try:
            want = bridge.validate(json.loads(json.dumps(exp)))
            mine = bridge.validate(json.loads(json.dumps(got)))
            r["serial"] = (json.dumps(want, ensure_ascii=False, default=str) ==
                           json.dumps(mine, ensure_ascii=False, default=str))
        except Exception:
            r["serial"] = None            # 기대 답 자체가 낡아 검증에 실패하면 셈에서 뺍니다

        exp_d, got_d = deltas(exp), deltas(got)
        if exp_d:
            r["direction"] = (len(got_d) == len(exp_d) and
                              all((a > 0) == (b > 0) for a, b in zip(exp_d, got_d)))
        if uses_move(exp):
            r["relative"] = uses_move(got)

    if any(g in instr for g in GESTURE):
        r["sequence"] = (len(got) >= 3)

    n = wanted_count(instr)
    if n is not None:
        r["count"] = (repeat_times(got) == n)

    return r


# ── 표 ──────────────────────────────────────────────────────────────────

def pct(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, 0
    return round(100 * sum(vals) / len(vals)), len(vals)


def print_table(results):
    models = list(results)
    w = max(18, max(len(m) for m in models) + 2)
    print("\n" + "─" * (26 + w * len(models)))
    print(f"  {'기준':<24}" + "".join(f"{m:>{w}}" for m in models))
    print("─" * (26 + w * len(models)))
    for key, name, _desc in CRITERIA:
        line = f"  {name:<24}"
        for m in models:
            p, n = pct(results[m]["grades"], key)
            line += f"{('—' if p is None else f'{p}% ({n})'):>{w}}"
        print(line)
    print("─" * (26 + w * len(models)))
    line = f"  {'평균 응답 시간':<24}"
    for m in models:
        ts = results[m]["times"]
        line += f"{(f'{sum(ts)/len(ts):.1f}초' if ts else '—'):>{w}}"
    print(line)
    line = f"  {'답을 못 낸 문제':<24}"
    for m in models:
        line += f"{results[m]['errors']:>{w}}"
    print(line)
    print("─" * (26 + w * len(models)) + "\n")
    print("  괄호 안은 그 기준이 해당되는 문제 수입니다. 해당 없는 문제는 평균에서 뺍니다.")


def main():
    ap = argparse.ArgumentParser(description="모델끼리 같은 문제로 겨루기")
    ap.add_argument("--models", nargs="+", required=True, help="Ollama 모델 이름들")
    ap.add_argument("--file", default=EVAL_FILE, help=f"평가 문제 파일 (기본 {EVAL_FILE})")
    ap.add_argument("--limit", type=int, help="앞에서 n문제만")
    ap.add_argument("--show-fails", action="store_true", help="틀린 답을 그대로 보여주기")
    ap.add_argument("--save", help="결과를 JSON으로 저장")
    ap.add_argument("--budget", type=int, default=1024,
                    help="한 답에 허용할 토큰 수 (기본 1024). 잘리면 올리세요")
    ap.add_argument("--think", action="store_true",
                    help="추론 모드를 켠 채로 재기 (기본은 꺼서 잼)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"⚠ {args.file} 가 없습니다.")
        print("  eval.jsonl 이나 arm_dataset_v3.jsonl 을 --file 로 지정하세요.")
        return 1

    problems = load_problems(args.file)
    if args.limit:
        problems = problems[:args.limit]
    have_exp = sum(1 for p in problems if p["expected"] is not None)
    print(f"\n  문제 {len(problems)}개 ({args.file}) · 기대 답이 있는 문제 {have_exp}개")
    print(f"  모델 {len(args.models)}개: {', '.join(args.models)}")
    print(f"  temperature 0 · 대화 기억 없음 · 토큰 예산 {args.budget}"
          f" · 추론 {'켬' if args.think else '끔'}\n")

    results = {}
    for model in args.models:
        print(f"  [{model}]", end="", flush=True)
        grades, times, fails, errors = [], [], [], 0
        for i, p in enumerate(problems, 1):
            got, dt, raw = ask(model, p["instruction"],
                               budget=args.budget, think=args.think)
            if got is None:
                errors += 1
                print("✘", end="", flush=True)
                fails.append({"instruction": p["instruction"], "why": raw})
                if "없음" in raw or "연결하지" in raw:
                    print(f"\n  ⚠ {raw}")
                    break
                continue
            times.append(dt)
            g = grade(p, got)
            grades.append(g)
            wrong = [k for k, v in g.items() if v is False]
            print("·" if not wrong else "×", end="", flush=True)
            if wrong:
                fails.append({"instruction": p["instruction"], "틀린 기준": wrong,
                              "모델 답": got, "기대": p["expected"]})
        print(f"  ({len(grades)}/{len(problems)})")
        results[model] = {"grades": grades, "times": times,
                          "errors": errors, "fails": fails}

    print_table(results)

    if args.show_fails:
        for model in args.models:
            fs = results[model]["fails"]
            if not fs:
                continue
            print(f"\n  ── {model} 이 틀린 것 {len(fs)}개 " + "─" * 30)
            for f in fs[:12]:
                print(f"\n  「{f['instruction']}」")
                if "why" in f:
                    print(f"    답을 못 냄: {f['why']}")
                    continue
                print(f"    틀린 기준: {', '.join(f['틀린 기준'])}")
                print(f"    모델 답: {json.dumps(f['모델 답'], ensure_ascii=False)[:180]}")
                if f["기대"]:
                    print(f"    기대   : {json.dumps(f['기대'], ensure_ascii=False)[:180]}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"file": args.file, "models": args.models,
                       "problems": len(problems), "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n  결과를 {args.save} 에 저장했습니다.")

    print("\n  읽는 법")
    print("    · valid 가 낮으면 그 모델은 형식을 못 지키는 것 — 프롬프트를 손봐야 합니다")
    print("    · ★ 실제 판단은 '팔의 동작이 같은가' 로 하세요. 표현이 달라도 팔이 같이 움직이면 맞은 것입니다")
    print("    · exact 는 참고용입니다. 같은 동작을 달리 표현한 것도 불일치로 세니까요")
    print("    · 한국어를 얼마나 알아듣는지는 숫자로 안 나옵니다.")
    print("      --show-fails 로 틀린 답을 직접 읽어 보세요. 그게 가장 정확합니다.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
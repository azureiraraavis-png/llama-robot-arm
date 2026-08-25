# -*- coding: utf-8 -*-
"""
evaluate.py — 학습 전/후 성적표

같은 평가 문제(학습에 쓰지 않은 15건)를 두 모델에 풀려서 채점한다.
  python evaluate.py --base            # 기존 Ollama 모델 (학습 전)
  python evaluate.py --tuned           # 학습된 LoRA (학습 후)
  python evaluate.py --base --tuned    # 둘 다 풀고 나란히 비교

채점 기준
  valid     : 브리지가 실행할 수 있는 형태인가
  exact     : 정답과 완전히 일치하는가 (가장 엄격)
  count     : "n번" 지시에서 실제 반복 횟수가 맞는가
  direction : 앞/뒤 방향 규약을 지키는가
  relative  : "n도 기울여"에 상대이동(move)을 쓰는가
  sequence  : 제스처를 단발 명령으로 때우지 않는가
"""

import argparse
import json
import re
import sys
from collections import defaultdict

from arm_common import SHORT_SYSTEM, score_one, to_messages

OLLAMA_URL = "http://localhost:11434/api/chat"


def parse_output(text: str):
    """모델 출력에서 명령 리스트를 뽑아낸다. 실패하면 None."""
    if not text:
        return None
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)  # 앞뒤 잡담 제거 시도
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        data = data.get("commands", [data])
    return data if isinstance(data, list) else None


# ── 학습 전: Ollama의 기본 모델 ─────────────────────────────────────────
def run_base(rows, model, system):
    import requests
    preds = []
    for i, r in enumerate(rows, 1):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": r["instruction"]}],
                "stream": False, "format": "json",
                # 채점은 재현 가능해야 한다 — temperature 0 + 고정 시드로 매번 같은 답을 받는다
                "options": {"temperature": 0.0, "top_p": 1.0, "seed": 42,
                            "num_ctx": 8192, "repeat_penalty": 1.0},
            }, timeout=300)
            preds.append(parse_output(resp.json()["message"]["content"]))
        except Exception as e:
            print(f"  ⚠ {i}번 실패: {e}")
            preds.append(None)
        print(f"\r  기본 모델 {i}/{len(rows)}", end="", flush=True)
    print()
    return preds


# ── 학습 후: 로컬 LoRA 어댑터 ───────────────────────────────────────────
def run_tuned(rows, adapter, base_model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    preds = []
    for i, r in enumerate(rows, 1):
        # 버전에 따라 반환형이 달라지므로 템플릿은 문자열로만 받고 따로 토큰화한다
        text_in = tok.apply_chat_template(
            to_messages(r["instruction"]), tokenize=False, add_generation_prompt=True)
        enc = tok(text_in, add_special_tokens=False, return_tensors="pt").to(model.device)
        n_in = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        preds.append(parse_output(tok.decode(out[0][n_in:], skip_special_tokens=True)))
        print(f"\r  학습 모델 {i}/{len(rows)}", end="", flush=True)
    print()
    return preds


def grade(rows, preds, label):
    totals, hits = defaultdict(int), defaultdict(int)
    failures = []
    for r, p in zip(rows, preds):
        ref = r["output"]["commands"]
        s = score_one(r["instruction"], p, ref)  # p가 None이면 전 기준 오답 처리
        for k, v in s.items():
            totals[k] += 1
            hits[k] += bool(v)
        if p is None:
            failures.append((r["instruction"], "JSON 파싱 실패"))
        elif not s.get("exact"):
            wrong = [k for k, v in s.items() if not v and k != "exact"]
            failures.append((r["instruction"],
                             ("위반: " + ", ".join(wrong)) if wrong else "정답과 다름(형식은 유효)"))
    return totals, hits, failures


def show(name, totals, hits):
    print(f"\n── {name} ──")
    order = ["valid", "exact", "count", "direction", "relative", "sequence"]
    for k in order:
        if totals.get(k):
            pct = hits[k] / totals[k] * 100
            bar = "█" * round(pct / 10) + "░" * (10 - round(pct / 10))
            # 문제 수가 적은 기준은 % 하나로 판단하면 위험하다
            note = "  ← 문제 수가 적어 참고용" if totals[k] < 5 else ""
            print(f"  {k:<10} {bar} {hits[k]:>2}/{totals[k]:<2} ({pct:>3.0f}%){note}")
    return {k: (hits[k] / totals[k] if totals.get(k) else None) for k in order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--base", action="store_true", help="학습 전 Ollama 모델 평가")
    ap.add_argument("--tuned", action="store_true", help="학습된 LoRA 평가")
    ap.add_argument("--ollama-model", default="llama3.1:8b")
    ap.add_argument("--adapter", default="arm-lora")
    ap.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--long-prompt", help="학습 전 평가에 쓸 긴 프롬프트 파일(선택)")
    ap.add_argument("--dump", help="문제별 정답/예측을 이 파일에 저장 (직접 눈으로 비교용)")
    args = ap.parse_args()

    if not (args.base or args.tuned):
        print("--base 또는 --tuned 중 하나 이상을 지정하세요")
        return

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    print(f"[평가 문제] {len(rows)}건 (학습에 쓰이지 않은 보류 세트)")

    results, preds_all = {}, {}
    if args.base:
        system = SHORT_SYSTEM
        if args.long_prompt:
            system = open(args.long_prompt, encoding="utf-8").read()
            print("  (기본 모델은 긴 프롬프트로 평가 — 공정한 비교를 위해)")
        p = run_base(rows, args.ollama_model, system)
        preds_all["base"] = p
        t, h, f = grade(rows, p, "기본")
        results["학습 전"] = show(f"학습 전 ({args.ollama_model})", t, h)
        if f:
            print("  틀린 문제:")
            for inst, why in f[:8]:
                print(f"    · {inst[:38]:<38} {why}")

    if args.tuned:
        p = run_tuned(rows, args.adapter, args.base_model)
        preds_all["tuned"] = p
        t, h, f = grade(rows, p, "학습")
        results["학습 후"] = show("학습 후 (arm-lora)", t, h)
        if f:
            print("  틀린 문제:")
            for inst, why in f[:8]:
                print(f"    · {inst[:38]:<38} {why}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fp:
            for i, r in enumerate(rows):
                fp.write(json.dumps({
                    "instruction": r["instruction"],
                    "정답": r["output"]["commands"],
                    "학습전": preds_all.get("base", [None] * len(rows))[i],
                    "학습후": preds_all.get("tuned", [None] * len(rows))[i],
                }, ensure_ascii=False) + "\n")
        print(f"\n[저장] 문제별 비교 → {args.dump}")

    if len(results) == 2:
        print("\n" + "=" * 46)
        print(f"{'기준':<12}{'학습 전':>10}{'학습 후':>10}{'변화':>12}")
        print("=" * 46)
        for k in ["valid", "exact", "count", "direction", "relative", "sequence"]:
            a, b = results["학습 전"].get(k), results["학습 후"].get(k)
            if a is None or b is None:
                continue
            d = (b - a) * 100
            mark = "▲" if d > 0 else ("▼" if d < 0 else "―")
            print(f"{k:<12}{a*100:>9.0f}%{b*100:>9.0f}%{mark:>6}{abs(d):>5.0f}%p")


if __name__ == "__main__":
    main()
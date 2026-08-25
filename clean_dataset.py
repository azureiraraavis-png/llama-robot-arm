# -*- coding: utf-8 -*-
"""
clean_dataset.py — arm_dataset.jsonl 정제기

수집된 데이터에는 시스템을 개선해가는 과정에서 생긴 모순이 섞여 있다.
(예: 같은 "뒤로 15도 기울여"가 어떤 날은 절대각 105, 어떤 날은 상대이동 +15)
모순된 데이터로 학습하면 모델은 아무것도 확실히 배우지 못한다.
이 스크립트는 명시적 규칙으로 표기를 통일하고, 명백한 오류를 제거한다.

사용: python clean_dataset.py arm_dataset.jsonl
출력: arm_dataset_clean.jsonl (정제본), clean_report.txt (변경 내역)
"""

import json
import re
import sys
from collections import defaultdict

SHOULDER, ELBOW, WRIST, GRIP = 1, 2, 3, 4
BEND_JOINTS = [SHOULDER, ELBOW, WRIST]  # "모든 관절 굽히기"의 대상 (베이스 회전·집게 제외)

report = []


def log(idx, what, detail=""):
    report.append(f"[{idx:>3}] {what}{(' — ' + detail) if detail else ''}")


# ── 규칙 1: 어깨 방향 규약 (앞으로 = 90 미만, 뒤로 = 90 초과) ──────────────
def fix_direction(idx, inst, cmds):
    fwd_first = "앞으로" in inst and (
        "뒤로" not in inst or inst.index("앞으로") < inst.index("뒤로")
    )
    if "앞으로" not in inst and "뒤로" not in inst:
        return cmds
    shoulder_moves = [c for c in cmds if c.get("cmd") == "joint" and c.get("joint") == SHOULDER]
    if not shoulder_moves:
        return cmds
    # 문장에 등장하는 방향 순서대로 어깨 명령에 기대 방향을 대응시킨다
    if "앞으로" in inst and "뒤로" in inst:
        expected = [fwd_first, not fwd_first]
    else:
        expected = [("앞으로" in inst)] * len(shoulder_moves)
    changed = False
    for c, want_fwd in zip(shoulder_moves, expected):
        a = c["angle"]
        if a == 90:
            continue
        is_fwd = a < 90
        if is_fwd != want_fwd:
            c["angle"] = 180 - a  # 90도 기준 대칭 반전
            changed = True
    if changed:
        log(idx, "어깨 방향 반전 교정", f"{[c['angle'] for c in shoulder_moves]}")
    return cmds


# ── 규칙 2: "n도(씩)" 증감 지시 → 절대각(joint) 대신 상대이동(move) ────────
REL_VERB = re.compile(r"(기울|굽히|접|숙이|올리|내리|돌리|펴)")
DEG = re.compile(r"(-?\d+)\s*도")


def to_relative(idx, inst, cmds):
    if not REL_VERB.search(inst):
        return cmds
    m = DEG.search(inst)
    if not m:
        return cmds
    deg = int(m.group(1))
    joints = [c for c in cmds if c.get("cmd") == "joint" and c.get("joint") != GRIP]
    if not joints or any(c.get("cmd") == "move" for c in cmds):
        return cmds

    # 방향 결정: 굽히다/접다/숙이다/앞으로 = 음수, 펴다/뒤로 = 양수. 지시문의 음수는 방향 반전.
    sign = -1 if re.search(r"(굽히|접|숙이|앞으로|내리)", inst) else 1
    if "뒤로" in inst or "올리" in inst:
        sign = 1
    delta = sign * abs(deg) * (-1 if deg < 0 else 1)

    if "모든 관절" in inst:
        targets = BEND_JOINTS
    else:
        targets = [c["joint"] for c in joints]

    rebuilt = []
    for c in cmds:
        if c.get("cmd") == "joint" and c.get("joint") != GRIP:
            continue
        rebuilt.append(c)
    moves = [{"cmd": "move", "joint": j, "delta": delta} for j in dict.fromkeys(targets)]
    # 원래 joint 명령이 있던 자리(맨 앞)에 move를 넣는다
    cmds = moves + rebuilt
    log(idx, f"절대각 → 상대이동({delta:+d}도)", f"관절 {list(dict.fromkeys(targets))}")
    return cmds


# ── 규칙 3: "n번" 반복 → repeat times로 압축 ────────────────────────────
CYCLE_N = re.compile(r"(\d+)\s*번")


def to_repeat_times(idx, inst, cmds):
    m = CYCLE_N.search(inst)
    if not m or any(c.get("cmd") == "repeat" for c in cmds):
        return cmds
    n = int(m.group(1))
    if n < 2:
        return cmds
    body = [c for c in cmds if c.get("cmd") != "home"]
    if not body:
        return cmds
    # 마지막 사이클의 꼬리 wait가 생략되어 균등 분할이 안 되는 경우를 보정
    if len(body) % n != 0 and body and body[0].get("cmd") != "wait":
        for pad in range(1, 3):
            if (len(body) + pad) % n == 0:
                body = body + [{"cmd": "wait", "ms": 200}] * pad
                break
    # 반복 단위 추정: 전체를 n등분해서 각 조각이 동일하면 그것이 단위
    if len(body) % n != 0:
        return cmds
    size = len(body) // n
    unit = body[:size]
    key = json.dumps(unit, ensure_ascii=False, sort_keys=True)
    for k in range(1, n):
        if json.dumps(body[k * size:(k + 1) * size], ensure_ascii=False, sort_keys=True) != key:
            return cmds  # 반복 구조가 아니면 건드리지 않음
    unit = [c for c in unit if c.get("cmd") != "wait"]  # 타이밍용 wait는 repeat가 대신 처리
    if not unit:
        return cmds
    tail = [c for c in cmds if c.get("cmd") == "home"]
    log(idx, f"수동 나열 {len(body)}개 → repeat times={n}")
    return [{"cmd": "repeat", "times": n, "commands": unit}] + tail[:1]


# ── 규칙 4: 자세 유지에 repeat 오용 → joint + wait ──────────────────────
def fix_hold(idx, inst, cmds):
    out = []
    changed = False
    for c in cmds:
        if c.get("cmd") == "repeat" and "seconds" in c and len(c.get("commands", [])) == 1:
            out.append(c["commands"][0])
            out.append({"cmd": "wait", "ms": c["seconds"] * 1000})
            changed = True
        else:
            out.append(c)
    if changed:
        log(idx, "자세 유지: repeat 오용 → joint + wait")
    return out


# ── 규칙 5: 유지 지시인데 home으로 되돌리는 모순 제거 ─────────────────────
HOLD = re.compile(r"(그대로|유지|원상복구 하지|복구하지|멈추시오|계속)")


def strip_trailing_home(idx, inst, cmds):
    holds = HOLD.search(inst) and "원상복구 하시오" not in inst
    relative = any(c.get("cmd") == "move" for c in cmds)
    if (holds or relative) and cmds and cmds[-1].get("cmd") == "home":
        log(idx, "유지/상대이동 지시의 끝 home 제거")
        return cmds[:-1]
    return cmds


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "arm_dataset.jsonl"
    rows = []
    for i, line in enumerate(open(src, encoding="utf-8"), 1):
        line = line.strip()
        if line:
            rows.append((i, json.loads(line)))

    # 명백히 잘못된 항목 (검증에서 확인된 것)
    DROP = {
        15: "지시문이 동작을 서술하지 않음(가르치는 말)",
        55: "'2번'인데 열기 1회 — 횟수 불일치",
        56: "'3번'인데 열기 2회 — 횟수 불일치",
        46: "무의미한 관절1=90 포함",
        47: "무의미한 관절1=90 포함",
        41: "'허리'를 관절0에 매핑(다른 항목은 관절1) — 모순",
    }

    cleaned = []
    seen = set()
    home_only_kept = 0

    for idx, r in rows:
        if idx in DROP:
            log(idx, "제거", DROP[idx])
            continue
        inst = r["instruction"].strip()
        cmds = json.loads(json.dumps(r["output"]["commands"]))  # 깊은 복사

        cmds = fix_direction(idx, inst, cmds)
        cmds = to_relative(idx, inst, cmds)
        cmds = to_repeat_times(idx, inst, cmds)
        cmds = fix_hold(idx, inst, cmds)
        cmds = strip_trailing_home(idx, inst, cmds)

        if not cmds:
            log(idx, "제거", "명령이 비었음")
            continue

        # home 단독 항목은 3건만 남긴다 (13건은 과대표집)
        if cmds == [{"cmd": "home"}]:
            home_only_kept += 1
            if home_only_kept > 3:
                log(idx, "제거", "home 단독 과대표집")
                continue

        key = (inst, json.dumps(cmds, ensure_ascii=False, sort_keys=True))
        if key in seen:
            log(idx, "제거", "완전 중복")
            continue
        seen.add(key)
        cleaned.append({"instruction": inst, "output": {"commands": cmds}})

    # 남은 모순(같은 지시문, 다른 출력) 확인 — 마지막 것만 남긴다 (최신 설계 반영)
    by_inst = defaultdict(list)
    for e in cleaned:
        by_inst[e["instruction"]].append(json.dumps(e["output"], ensure_ascii=False, sort_keys=True))
    conflict = {k for k, v in by_inst.items() if len(set(v)) > 1}
    if conflict:
        final, kept = [], {}
        for e in reversed(cleaned):
            if e["instruction"] in conflict:
                if e["instruction"] in kept:
                    continue
                kept[e["instruction"]] = True
            final.append(e)
        final.reverse()
        for c in conflict:
            log(0, "모순 해소", f"'{c[:40]}' — 최신 항목만 유지")
        cleaned = final

    with open("arm_dataset_clean.jsonl", "w", encoding="utf-8") as f:
        for e in cleaned:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    header = [
        "=" * 60,
        f"원본 {len(rows)}건 → 정제본 {len(cleaned)}건",
        "=" * 60,
        "",
    ]
    with open("clean_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(header + report) + "\n")

    print("\n".join(header + report))


if __name__ == "__main__":
    main()
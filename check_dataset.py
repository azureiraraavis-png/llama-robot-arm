# -*- coding: utf-8 -*-
"""
check_dataset.py (v2) — 학습 전 데이터 검사기

모순된 데이터로 학습하면 모델은 아무것도 확실히 배우지 못한다.
1차 검사기는 부호·방향만 봤는데, 그 뒤 home 정책과 시간 환산에서도
같은 종류의 사고가 나서 검사 항목을 넓혔다.

검사 항목
  1. 모순      같은 지시문에 다른 정답
  2. 부호      "n도씩 굽히다"=음수 / "-n도씩"=양수(뒤로)
  3. 방향      앞으로=90 미만 / 뒤로=90 초과 (어깨)
  4. home 정책 복구 명시↔home / 유지 명시↔home 없음 / n초 대기↔home
  5. 시간      지시문의 "n초"와 계획의 대기 시간 일치
  6. 대상      "모든 관절"에 집게(관절4)가 섞이는지
  7. 형식      알 수 없는 명령, 관절 번호 범위

사용: python check_dataset.py arm_dataset_v3.jsonl
"""

import json
import re
import sys
from collections import defaultdict

SHOULDER = 1
OK_CMDS = {"home", "joint", "move", "grip", "wait", "hold", "speed", "repeat", "pose"}

BEND = re.compile(r"(굽히|접|숙이)")
DEG = re.compile(r"(-?\d+)\s*도")
DURATION = re.compile(r"(\d+)\s*초")

# home 규약은 브리지에 단 하나만 정의돼 있다 (규칙이 두 곳에 복사되면 어긋난다)
try:
    from llm_arm_bridge import home_policy as expected_home
except ImportError:
    print("⚠ llm_arm_bridge.py를 찾을 수 없어 home 검사는 건너뜁니다 "
          "(같은 폴더에 두세요)")
    def expected_home(_inst):
        return None


def expected_sign(inst):
    m = DEG.search(inst)
    if not m:
        return None
    deg = int(m.group(1))
    if BEND.search(inst):
        base = -1
    elif "펴" in inst or "올리" in inst:
        base = +1
    elif "뒤로" in inst:
        return +1
    elif "앞으로" in inst or "내리" in inst:
        return -1
    else:
        return None
    return base * (-1 if deg < 0 else 1)


def duration_slots(cmds):
    out = []
    for c in cmds:
        if not isinstance(c, dict):
            continue
        if c.get("cmd") in ("wait", "hold"):
            out.append(int(float(c["seconds"]) * 1000) if "seconds" in c else int(c.get("ms", 0)))
        elif c.get("cmd") == "repeat":
            if "seconds" in c:
                out.append(int(c["seconds"]) * 1000)
            out += duration_slots(c.get("commands", []))
    return out


def check_format(cmds, problems, inst):
    for c in cmds:
        if not isinstance(c, dict) or c.get("cmd") not in OK_CMDS:
            problems.append(("형식", f"'{inst[:36]}' 알 수 없는 명령: {c}"))
            continue
        if c.get("cmd") in ("joint", "move") and not (0 <= c.get("joint", -1) <= 4):
            problems.append(("형식", f"'{inst[:36]}' 관절 번호 {c.get('joint')}"))
        if c.get("cmd") == "repeat":
            check_format(c.get("commands", []), problems, inst)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "arm_dataset_v3.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    problems = []

    # 1. 모순
    by = defaultdict(set)
    for r in rows:
        by[r["instruction"]].add(json.dumps(r["output"], ensure_ascii=False, sort_keys=True))
    for inst, outs in by.items():
        if len(outs) > 1:
            problems.append(("모순", f"'{inst[:40]}' — 정답 {len(outs)}종"))

    for r in rows:
        inst, cmds = r["instruction"], r["output"]["commands"]

        # 2. 부호
        want = expected_sign(inst)
        if want is not None:
            for c in cmds:
                if c.get("cmd") == "move" and c.get("delta"):
                    if (1 if c["delta"] > 0 else -1) != want:
                        problems.append(("부호", f"'{inst[:36]}' delta={c['delta']:+d}"))

        # 3. 방향
        if ("앞으로" in inst) ^ ("뒤로" in inst):
            fwd = "앞으로" in inst
            for c in cmds:
                if c.get("cmd") == "joint" and c.get("joint") == SHOULDER:
                    a = c.get("angle", 90)
                    if a != 90 and (a < 90) != fwd:
                        problems.append(("방향", f"'{inst[:36]}' 관절1={a}"))

        # 4. home 정책
        wh = expected_home(inst)
        if wh is not None and cmds:
            has = cmds[-1].get("cmd") == "home"
            if wh and not has:
                problems.append(("home", f"'{inst[:40]}' 복귀해야 하는데 home 없음"))
            elif has and not wh:
                problems.append(("home", f"'{inst[:40]}' 유지해야 하는데 home 있음"))

        # 5. 시간
        wants = [int(x) * 1000 for x in DURATION.findall(inst)]
        slots = [s for s in duration_slots(cmds) if s >= 1000]
        if wants and len(wants) == len(slots) and sorted(wants) != sorted(slots):
            problems.append(("시간", f"'{inst[:36]}' 지시 {[w//1000 for w in wants]}초 / "
                                     f"계획 {[s/1000 for s in slots]}초"))

        # 6. 대상
        if "모든 관절" in inst and "servo5" not in inst:
            if any(c.get("cmd") == "joint" and c.get("joint") == 4 for c in cmds):
                problems.append(("대상", f"'{inst[:40]}' 집게(관절4)가 포함됨"))

        # 7. 형식
        check_format(cmds, problems, inst)

    print(f"검사 대상 {len(rows)}건")
    if not problems:
        print("✅ 문제 없음 — 학습해도 좋습니다")
        return 0
    print(f"⚠ 문제 {len(problems)}건:\n")
    for kind, msg in problems:
        print(f"  [{kind:<3}] {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
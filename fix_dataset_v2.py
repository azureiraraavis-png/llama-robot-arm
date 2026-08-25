# -*- coding: utf-8 -*-
"""
fix_dataset_v2.py — 1차 정제에서 놓친 모순만 골라서 교정

원칙: 멀쩡한 항목은 건드리지 않는다. 실제로 모순인 것만 고친다.
(1차 정제가 규칙을 광범위하게 적용하다 사고를 냈으므로, 이번엔 외과적으로 간다)

교정 대상 4가지
  A. "n초 기다리/대기/가만히 있" (계속 제외) → 끝에 home  [사용자 확정 규약: 시간 대기 후 복귀]
  B. "원상복구/원상복귀/원래 상태로 돌아" 명시 → 끝에 home
  C. "원상복구 하지 마/하지 말" 명시          → 끝 home 제거
  D. "모든 관절"(servo5 표현 제외)의 대상에서 집게(관절4) 제외

건드리지 않는 것
  - 자세를 잡는 지시("집게를 닫으시오", "팔을 C형으로 굽히시오") — home을 붙이면 자세가 사라진다
  - "유지하시오 / 멈추시오 / 계속" — 무기한 유지 의도
  - 상대이동만 있는 지시 — home을 붙이면 이동이 즉시 사라진다

사용: python fix_dataset_v2.py arm_dataset_clean_fixed.jsonl
출력: arm_dataset_v3.jsonl, fix_v2_report.txt
"""

import json
import re
import sys

DURATION = re.compile(r"\d+\s*초")
WAIT_VERB = re.compile(r"(기다리|대기|가만히\s*있)")
FOREVER = re.compile(r"계속")
NO_RETURN = re.compile(r"(원상복구|원상복귀|복구)\s*하지\s*(마|말)")
RETURN = re.compile(r"(원상복구|원상복귀|원래\s*상태로\s*돌아|원위치로\s*되돌)")

report = []


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "arm_dataset_clean_fixed.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    changed = 0

    for r in rows:
        inst = r["instruction"]
        cmds = r["output"]["commands"]
        has_home = bool(cmds) and cmds[-1].get("cmd") == "home"

        # ── D. "모든 관절"에서 집게 제외 ─────────────────────────────
        if "모든 관절" in inst and "servo5" not in inst:
            n0 = len(cmds)
            cmds = [c for c in cmds if not (c.get("cmd") == "joint" and c.get("joint") == 4)]
            if len(cmds) != n0:
                report.append(f"[D 집게제외] {inst[:46]}")
                changed += 1
            has_home = bool(cmds) and cmds[-1].get("cmd") == "home"

        # ── E. 1초 이상 대기는 seconds 표기로 (모델이 곱셈하지 않도록) ──
        for c in cmds:
            if c.get("cmd") == "wait" and "ms" in c and c["ms"] >= 1000:
                secs = round(c["ms"] / 1000)
                c.pop("ms")
                c["seconds"] = secs
                report.append(f"[E 초표기] {inst[:40]} → {secs}초")
                changed += 1

        # ── C. 복구 금지 명시 → home 제거 ───────────────────────────
        if NO_RETURN.search(inst):
            if has_home:
                cmds = cmds[:-1]
                report.append(f"[C home제거] {inst[:46]}")
                changed += 1
        # ── B. 복구 명시 → home 보장 ───────────────────────────────
        elif RETURN.search(inst):
            if not has_home:
                cmds = cmds + [{"cmd": "home"}]
                report.append(f"[B home추가] {inst[:46]}")
                changed += 1
        # ── A. 시간 대기 → home 보장 ───────────────────────────────
        elif DURATION.search(inst) and WAIT_VERB.search(inst) and not FOREVER.search(inst):
            if not has_home:
                cmds = cmds + [{"cmd": "home"}]
                report.append(f"[A home추가] {inst[:46]}")
                changed += 1

        r["output"]["commands"] = cmds

    with open("arm_dataset_v3.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    head = ["=" * 58, f"{len(rows)}건 중 {changed}곳 교정", "=" * 58, ""]
    open("fix_v2_report.txt", "w", encoding="utf-8").write("\n".join(head + report) + "\n")
    print("\n".join(head + report))


if __name__ == "__main__":
    main()
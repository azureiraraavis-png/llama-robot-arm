# -*- coding: utf-8 -*-
"""
checkup.py — 이 폴더가 온전한 작업공간인지 확인합니다

작업공간을 옮긴 뒤에 돌리세요. 무엇이 있고 무엇이 빠졌는지, 빠진 것이
어느 기능을 못 쓰게 만드는지 알려줍니다.

  py -3.12 checkup.py           빠른 확인 (파일과 설정만)
  py -3.12 checkup.py --deep    불러오기까지 시험 (torch·mediapipe가 뜨는지)
"""

import argparse
import glob
import json
import os
import sys

OK, MISS, WARN = "✔", "✘", "△"

# 기능별로 무엇이 필요한지. (파일, 없으면 치명적인가)
GROUPS = [
    ("팔 조종 (자연어)", [
        ("llm_arm_bridge.py", True), ("dance.py", True),
    ]),
    ("직접 구운 모델", [
        ("arm_tuned.py", True), ("arm_common.py", True), ("evaluate.py", False),
    ]),
    ("카메라 · 가리키기", [
        ("vision.py", True), ("point_at.py", True),
        ("calibrate_point.py", False), ("point_calib.json", True),
        ("vision_config.json", False), ("vision_check.py", False),
    ]),
    ("손 추적", [
        ("hand_map.py", True), ("hand_track.py", True),
        ("hand_landmarker.task", False), ("mp_check.py", False),
    ]),
    ("음성", [
        ("voice_io.py", True), ("stt.py", True),
        ("voice_arm.py", True), ("voice_check.py", False),
    ]),
    ("데이터 · 기록", [
        ("check_dataset.py", False), ("대장정_기록.md", False),
    ]),
]

IMPORTS = {
    "llm_arm_bridge": "팔 조종",
    "dance": "춤",
    "hand_map": "손 → 관절 변환",
    "voice_io": "마이크",
    "vision": "카메라",
}


def find_adapter():
    """PEFT 어댑터 폴더는 adapter_config.json 으로 알아봅니다."""
    hits = glob.glob("*/adapter_config.json") + glob.glob("*/*/adapter_config.json")
    return sorted({os.path.dirname(h) for h in hits})


def find_sketches():
    return sorted(glob.glob("arduino/*/*.ino") + glob.glob("*/*.ino") + glob.glob("*.ino"))


def check_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="작업공간 점검")
    ap.add_argument("--deep", action="store_true", help="모듈 불러오기까지 시험")
    args = ap.parse_args()

    print(f"\n  여기: {os.getcwd()}")
    print(f"  파이썬 {sys.version.split()[0]}\n")

    fatal, minor = [], []

    for title, items in GROUPS:
        lines = []
        for name, needed in items:
            if os.path.exists(name):
                extra = ""
                if name.endswith(".json"):
                    err = check_json(name)
                    if err:
                        extra = f"  ← 내용이 깨졌습니다 ({err})"
                        minor.append(f"{name} 내용 확인 필요")
                size = os.path.getsize(name)
                unit = f"{size/1024/1024:.1f}MB" if size > 1024 * 1024 else f"{size/1024:.0f}KB"
                lines.append(f"    {OK} {name:<26} {unit:>7}{extra}")
            else:
                mark = MISS if needed else WARN
                lines.append(f"    {mark} {name:<26} 없음")
                (fatal if needed else minor).append(f"{name} ({title})")
        print(f"  {title}")
        print("\n".join(lines))
        print()

    # 학습된 모델
    print("  학습된 모델 (arm-lora)")
    adapters = find_adapter()
    if adapters:
        for d in adapters:
            n = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(d) for f in fs)
            unit = f"{n/1024/1024:.0f}MB" if n > 1024 * 1024 else f"{n/1024:.0f}KB"
            print(f"    {OK} {d:<26} {unit:>7}")
            if n < 1024 * 1024:
                print(f"      {WARN} 어댑터치고 너무 작습니다 — 파일이 덜 왔을 수 있습니다")
                minor.append(f"{d} 크기 확인")
    else:
        print(f"    {MISS} adapter_config.json 이 있는 폴더를 찾지 못했습니다")
        fatal.append("학습된 모델 폴더 (arm_tuned.py를 못 씁니다)")
    print()

    # 아두이노
    print("  아두이노 스케치")
    sk = find_sketches()
    if sk:
        for s in sk:
            print(f"    {OK} {s}")
    else:
        print(f"    {WARN} .ino 를 찾지 못했습니다 (팔은 이미 구워져 있으면 당장은 문제없음)")
        minor.append("아두이노 스케치")
    print()

    # 데이터셋
    ds = sorted(glob.glob("*.jsonl"))
    print("  데이터셋")
    if ds:
        for d in ds:
            with open(d, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            print(f"    {OK} {d:<26} {n}건")
    else:
        print(f"    {WARN} .jsonl 이 없습니다")
        minor.append("데이터셋")
    print()

    if args.deep:
        print("  불러오기 시험")
        sys.path.insert(0, os.getcwd())
        for mod, what in IMPORTS.items():
            try:
                __import__(mod)
                print(f"    {OK} {mod:<26} {what}")
            except Exception as e:
                msg = str(e).splitlines()[0][:70]
                print(f"    {MISS} {mod:<26} {type(e).__name__}: {msg}")
                fatal.append(f"{mod} 불러오기 실패")
        for mod, hint in (("torch", "직접 구운 모델 + 음성인식"),
                          ("mediapipe", "손 추적"),
                          ("serial", "아두이노 통신"),
                          ("cv2", "카메라")):
            try:
                m = __import__(mod)
                v = getattr(m, "__version__", "")
                print(f"    {OK} {mod:<26} {v}  ({hint})")
            except Exception as e:
                msg = str(e).splitlines()[0][:60]
                critical = (mod == "serial")      # 이게 없으면 팔이 아예 안 움직입니다
                print(f"    {MISS if critical else WARN} {mod:<26} {type(e).__name__}: {msg}")
                (fatal if critical else minor).append(f"{mod} — {hint}")
        print()

    print("─" * 60)
    if not fatal and not minor:
        print("  전부 제자리에 있습니다. 그대로 쓰시면 됩니다.")
    if fatal:
        print("  빠지면 안 되는 것:")
        for f in fatal:
            print(f"    {MISS} {f}")
        print("\n  원본 폴더(Downloads)에서 직접 가져오세요.")
    if minor:
        print("  없어도 당장은 되는 것:")
        for m in minor:
            print(f"    {WARN} {m}")
    print()
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
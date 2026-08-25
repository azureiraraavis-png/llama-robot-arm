# -*- coding: utf-8 -*-
"""
dance.py — 로봇팔 안무

설계 두 가지
  1. 관절을 하나씩(S) 말고 전체 자세(P)로 움직인다.
     펌웨어의 moveTo가 모든 관절을 동시에 목표로 몰아가므로 동작이 흐른다.
  2. 박자에 맞추기 위해 속도를 역산한다.
     펌웨어는 1도당 (22 - speed*2)ms 로 움직인다. 한 박자에 D도를 움직여야 하면
     필요한 ms/도 = beat/D 이고, 거기서 speed를 거꾸로 구한다.
     그러면 큰 동작도 작은 동작도 같은 박자에 떨어지고 급가속이 생기지 않는다.

이 팔은 레이저 커팅 나무에 마이크로 서보다. 관절 한계를 꽉 채워 쓰지 않고
안쪽으로 여유를 둔 '춤 전용 범위'를 쓴다. 무리한 자세로 서보를 갈지 않기 위함이다.

실행
  py -3.12 dance.py                    # 전곡 (모든 안무 이어서)
  py -3.12 dance.py --list             # 안무 목록
  py -3.12 dance.py -r wave -r bow     # 고른 안무만
  py -3.12 dance.py --bpm 140          # 빠르게 (기본 100)
  py -3.12 dance.py --loop 3           # 3번 반복
  py -3.12 dance.py --amp 0.6          # 동작을 작게 (팔이 흔들릴 때)
  py -3.12 dance.py --no-grip          # 집게 동작 빼기 (그리퍼 선이 빠질 때)
  py -3.12 dance.py --dry-run          # 팔 없이 예상 동작·시간만
"""

import argparse
import sys
import time

# 자세 = [베이스, 어깨, 팔꿈치, 손목, 그리퍼]
HOME = [90, 90, 90, 90, 60]

# 춤 전용 안전 범위 (하드웨어 한계보다 안쪽으로 여유를 둔다)
# 손목(관절3)은 범위를 좁게 잡는다 — 그리퍼 서보 선이 손목 회전에 따라 비틀리기 때문.
# 실제로 춤 도중 그리퍼 커넥터가 빠지는 일이 있어 145도→118도로 줄였다.
SAFE = [(35, 145), (50, 130), (50, 130), (62, 118), (55, 115)]

MAX_STEP = 65      # 한 박자에 이보다 큰 각도 이동은 금지 (급가속 방지)


def clamp(pose):
    return [max(lo, min(hi, int(v))) for v, (lo, hi) in zip(pose, SAFE)]


def scale(pose, amp):
    """자세를 홈 쪽으로 끌어당겨 진폭을 줄인다. amp=1.0이면 원래대로, 0.5면 절반."""
    if amp >= 0.999:
        return clamp(pose)
    return clamp([h + (v - h) * amp for v, h in zip(pose, HOME)])


# ── 안무 ────────────────────────────────────────────────────────────────
# 각 항목은 (자세, 박자수) 또는 ("grip", 닫힘여부, 박자수) 또는 ("rest", 박자수)
ROUTINES = {
    "bow": ("꾸벅 인사", [
        ([90, 65, 105, 90, 60], 1),
        ([90, 60, 110, 90, 60], 1),
        (HOME, 1),
        ("rest", 1),
    ]),
    "wave": ("웨이브 — 어깨에서 손목으로 물결", [
        ([90, 72, 96, 90, 60], 1),
        ([90, 92, 74, 100, 60], 1),
        ([90, 96, 96, 68, 60], 1),
        ([90, 90, 90, 112, 60], 1),
        (HOME, 1),
    ]),
    "sway": ("좌우 스윙 — 손목이 반대로 돈다", [
        ([60, 85, 96, 70, 60], 1),
        ([120, 85, 96, 112, 60], 1),
        ([60, 85, 96, 70, 60], 1),
        ([120, 85, 96, 112, 60], 1),
        (HOME, 1),
    ]),
    "shake": ("도리도리", [
        ([90, 80, 100, 68, 60], 1),
        ([90, 80, 100, 112, 60], 1),
        ([90, 80, 100, 68, 60], 1),
        ([90, 80, 100, 112, 60], 1),
        ([90, 80, 100, 90, 60], 1),
    ]),
    "clap": ("딱딱 박자 — 집게로 리듬", [
        ([90, 78, 104, 90, 60], 1),
        ("grip", True, 0.5),
        ("grip", False, 0.5),
        ("grip", True, 0.5),
        ("grip", False, 0.5),
        ("grip", True, 0.5),
        ("grip", False, 1),
    ]),
    "robot": ("로봇춤 — 각지게 끊어서", [
        ([70, 70, 112, 66, 60], 1),
        ("rest", 0.5),
        ([110, 112, 70, 114, 60], 1),
        ("rest", 0.5),
        ([70, 112, 112, 66, 60], 1),
        ("rest", 0.5),
        ([110, 70, 70, 114, 60], 1),
        ("rest", 0.5),
        (HOME, 1),
    ]),
    "sweep": ("큰 스윕 — 팔을 들고 좌우로", [
        ([90, 118, 70, 90, 60], 1),
        ([40, 118, 70, 66, 60], 2),
        ([140, 118, 70, 114, 60], 2),
        ([90, 118, 70, 90, 60], 1),
    ]),
    "finale": ("피날레 — 스윕, 딱딱, 깊은 인사", [
        ([40, 115, 72, 66, 60], 1),
        ([140, 115, 72, 114, 60], 1),
        ([90, 110, 80, 90, 60], 1),
        ("grip", True, 0.5),
        ("grip", False, 0.5),
        ([90, 58, 112, 90, 60], 2),
        ("rest", 1),
        (HOME, 1),
    ]),
}

ORDER = ["bow", "wave", "sway", "shake", "clap", "robot", "sweep", "finale"]

# 모델이나 사람이 한국어로 부를 수 있게
ALIASES = {
    "전체": "all", "다": "all", "춤": "all", "풀": "all",
    "인사": "bow", "꾸벅": "bow",
    "웨이브": "wave", "물결": "wave",
    "스윙": "sway", "좌우": "sway", "흔들": "sway",
    "도리도리": "shake", "고개": "shake",
    "박수": "clap", "딱딱": "clap", "리듬": "clap",
    "로봇": "robot", "각": "robot",
    "스윕": "sweep", "크게": "sweep",
    "피날레": "finale", "마무리": "finale",
}


def resolve(name):
    """안무 이름을 키로. 'all'이면 전곡, 모르면 None."""
    if not name:
        return ORDER
    n = str(name).strip().lower()
    if n in ("all", "전체", "*"):
        return list(ORDER)
    if n in ROUTINES:
        return [n]
    if n in ALIASES:
        a = ALIASES[n]
        return list(ORDER) if a == "all" else [a]
    for k, v in ALIASES.items():          # 부분 일치 (예: "웨이브 해줘")
        if k in n:
            return list(ORDER) if v == "all" else [v]
    return None


def to_lines(routine_keys, bpm=100, amp=1.0, use_grip=True):
    """안무를 아두이노 시리얼 명령 줄로 펼친다. (브리지가 이걸 받아 그대로 보낸다)"""
    beat_ms = int(60000 / max(40, min(200, int(bpm))))
    amp = max(0.2, min(1.0, float(amp)))
    out = []
    for st in plan(routine_keys, beat_ms, amp, use_grip):
        if st[0] == "pose":
            out.append(f"SPEED {st[2]}")
            out.append("P " + " ".join(str(v) for v in st[1]))
        elif st[0] == "grip":
            out.append(f"GRIP {1 if st[1] else 0}")
        elif st[0] == "wait":
            ms = int(st[1])
            while ms > 5000:                # 아두이노 1회 대기 상한
                out.append("WAIT 5000")
                ms -= 5000
            if ms > 0:
                out.append(f"WAIT {ms}")
    out.append("SPEED 5")
    return out


# ── 박자 ↔ 속도 ─────────────────────────────────────────────────────────
def speed_for(delta_deg, beat_ms):
    """이 이동을 beat_ms 안에 끝내려면 몇 단계 속도가 필요한가.
    펌웨어: 1도당 (22 - speed*2) ms → speed = (22 - ms_per_deg) / 2"""
    if delta_deg <= 0:
        return 10, 0
    ms_per_deg = beat_ms / delta_deg
    speed = int(round((22 - ms_per_deg) / 2))
    speed = max(1, min(10, speed))
    actual = delta_deg * (22 - speed * 2)      # 실제 걸릴 시간
    return speed, actual


def plan(routine_keys, beat_ms, amp=1.0, use_grip=True):
    """실행 전에 전체 계획을 만든다. (팔 없이도 확인할 수 있게)"""
    steps, cur = [], list(HOME)
    for key in routine_keys:
        title, moves = ROUTINES[key]
        steps.append(("title", key, title))
        for m in moves:
            if m[0] == "rest":
                steps.append(("wait", int(m[1] * beat_ms)))
            elif m[0] == "grip":
                _, close, beats = m
                if not use_grip:                      # 그리퍼 선이 불안할 때
                    steps.append(("wait", int(beats * beat_ms)))
                    continue
                steps.append(("grip", close))
                steps.append(("wait", int(beats * beat_ms)))
                cur[4] = 110 if close else 60
            else:
                pose, beats = scale(m[0], amp), m[1]
                delta = max(abs(a - b) for a, b in zip(pose, cur))
                if delta > MAX_STEP:                  # 급가속 방지: 반씩 나눠 간다
                    mid = clamp([(a + b) // 2 for a, b in zip(pose, cur)])
                    for p in (mid, pose):
                        d = max(abs(a - b) for a, b in zip(p, cur))
                        sp, took = speed_for(d, beats * beat_ms / 2)
                        steps.append(("pose", p, sp, took))
                        cur = list(p)
                else:
                    sp, took = speed_for(delta, beats * beat_ms)
                    steps.append(("pose", pose, sp, took))
                    pad = beats * beat_ms - took
                    if pad > 40:
                        steps.append(("wait", int(pad)))
                    cur = list(pose)
    steps.append(("pose", HOME, 5, 0))
    return steps


def run(steps, ser, dry):
    import llm_arm_bridge as bridge      # 순환 import를 피해 여기서 부른다
    beat = 0
    for st in steps:
        if st[0] == "title":
            print(f"\n♪ {st[1]} — {st[2]}")
        elif st[0] == "wait":
            if not dry:
                time.sleep(st[1] / 1000)
        elif st[0] == "grip":
            beat += 1
            print(f"  [{beat:>3}] 집게 {'닫기' if st[1] else '열기'}")
            if not dry:
                bridge.send_serial(ser, [f"GRIP {1 if st[1] else 0}"])
        else:
            _, pose, sp, took = st
            beat += 1
            print(f"  [{beat:>3}] 자세 {pose}  속도 {sp}  (약 {took}ms)")
            if not dry:
                bridge.send_serial(ser, [f"SPEED {sp}",
                                         "P " + " ".join(str(v) for v in pose)])


def main():
    ap = argparse.ArgumentParser(description="로봇팔 안무")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("-r", "--routine", action="append", help="안무 이름 (여러 번 지정 가능)")
    ap.add_argument("--bpm", type=int, default=100, help="분당 박자 (기본 100)")
    ap.add_argument("--loop", type=int, default=1, help="반복 횟수")
    ap.add_argument("--amp", type=float, default=1.0,
                    help="동작 크기 (1.0=원래, 0.6=작게). 팔이 흔들리거나 선이 걱정될 때")
    ap.add_argument("--no-grip", action="store_true",
                    help="집게 동작을 건너뛴다 (그리퍼 선이 빠질 때)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="팔 없이 계획만 확인")
    args = ap.parse_args()

    if args.list:
        print("안무 목록:")
        for k in ORDER:
            print(f"  {k:<8} {ROUTINES[k][0]}")
        return 0

    keys = args.routine or ORDER
    bad = [k for k in keys if k not in ROUTINES]
    if bad:
        print(f"⚠ 모르는 안무: {', '.join(bad)}  (--list 로 확인)")
        return 1

    beat_ms = int(60000 / max(40, min(200, args.bpm)))
    steps = plan(keys, beat_ms, max(0.2, min(1.0, args.amp)), not args.no_grip)
    total = sum(s[1] if s[0] == "wait" else (s[3] if s[0] == "pose" else 0) for s in steps)
    print(f"BPM {args.bpm} (한 박자 {beat_ms}ms) · 안무 {len(keys)}개 · "
          f"1회 약 {total/1000:.0f}초 · {args.loop}회 반복"
          + (f" · 진폭 {args.amp:.1f}" if args.amp < 0.999 else "")
          + (" · 집게 끔" if args.no_grip else ""))

    ser = None
    if not args.dry_run:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.5)
        time.sleep(2.5)
        ser.reset_input_buffer()
        print(f"[연결됨] {args.port}")
        bridge.send_serial(ser, ["SPEED 5", "HOME"])
        time.sleep(0.5)

    try:
        for i in range(args.loop):
            if args.loop > 1:
                print(f"\n=== {i+1}/{args.loop} 회 ===")
            run(steps, ser, args.dry_run)
    except KeyboardInterrupt:
        print("\n중단합니다.")
    finally:
        if ser:
            bridge.send_serial(ser, ["SPEED 5", "HOME"])
            ser.close()
    print("\n끝.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
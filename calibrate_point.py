# -*- coding: utf-8 -*-
"""
calibrate_point.py — "가리키기" 보정 (1차원)

집기 보정이 어려운 이유는 화면의 2차원 위치를 관절 여러 개에 대응시켜야 하고,
팔이 물체에 실제로 닿아야 하기 때문이다. 가리키기는 그 부담이 없다.
  · 물체는 바닥에 누워 있어도 된다 (집지 않으므로)
  · 팔은 정해진 '가리키는 자세'를 유지한 채 베이스만 돌린다
  · 화면 x → 베이스 각도, 1차원 대응
  · 단조성(화면 오른쪽으로 갈수록 각도가 한 방향)으로 검증할 수 있다

순서
  1. 팔을 '가리키는 자세'로 만든 뒤 [t] — 이 자세는 이후 고정된다
  2. 물체를 놓고 팔을 치운 뒤 [r] — 화면 위치 기억
  3. a/d 로 베이스만 돌려 물체를 가리키게 한 뒤 [스페이스]
  4. 물체를 화면 좌→우로 옮겨가며 5~7번 반복

실행: py -3.12 calibrate_point.py            (COM4, 0번 카메라)
      py -3.12 calibrate_point.py --fresh

조작
  t        현재 자세를 '가리키는 자세'로 고정
  r        물체 위치 기억      스페이스  베이스 각도 저장
  a/d      베이스 좌우         w/s q/e z/c  자세 조정(어깨·팔꿈치·손목)
  [ / ]    조작 단위           u  취소   h  홈   x  저장종료   ESC  취소종료
"""

import argparse
import json
import os
import sys
import time

import cv2

import llm_arm_bridge as bridge
import vision

OUT = "point_calib.json"
MIN_GAP_PX = 45          # 이보다 가까우면 같은 자리로 본다

JOG = {ord("a"): (0, -1), ord("d"): (0, +1),
       ord("s"): (1, -1), ord("w"): (1, +1),
       ord("q"): (2, -1), ord("e"): (2, +1),
       ord("z"): (3, -1), ord("c"): (3, +1)}


def check_monotonic(points):
    """화면 x 순으로 정렬했을 때 베이스 각도가 한 방향으로 가는지.
    반환: (문제없음?, 설명)"""
    TOL = 3                                  # 서보 유격을 감안한 여유
    if len(points) < 3:
        return True, "점 3개 이상부터 검사합니다"
    s = sorted(points, key=lambda p: p["px_x"])
    bases = [p["base"] for p in s]
    steps = [b - a for a, b in zip(bases, bases[1:])]

    rising = all(d >= -TOL for d in steps)    # 화면 오른쪽으로 갈수록 각도 증가
    falling = all(d <= TOL for d in steps)
    if rising or falling:
        return True, "단조 관계 ✓ (" + ("증가" if rising else "감소") + ")"

    # 전체 추세와 반대로 가는 구간을 짚어준다
    trend = 1 if bases[-1] >= bases[0] else -1
    breaks = [(s[i]["px_x"], bases[i], s[i + 1]["px_x"], bases[i + 1])
              for i, d in enumerate(steps) if d * trend < -TOL]
    if not breaks:                            # 추세 자체가 모호한 경우
        worst = max(range(len(steps)), key=lambda i: abs(steps[i]))
        breaks = [(s[worst]["px_x"], bases[worst],
                   s[worst + 1]["px_x"], bases[worst + 1])]
    return False, f"단조성 깨짐 {len(breaks)}곳: " + \
        ", ".join(f"x{a}({b}도)→x{c}({d}도)" for a, b, c, d in breaks[:3])


def predict(points, px_x):
    """보정점 사이를 선형 보간해 베이스 각도를 낸다. 범위 밖은 양 끝값으로 고정."""
    if not points:
        return None
    s = sorted(points, key=lambda p: p["px_x"])
    xs = [p["px_x"] for p in s]
    bs = [p["base"] for p in s]
    if px_x <= xs[0]:
        return bs[0]
    if px_x >= xs[-1]:
        return bs[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= px_x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return bs[i]
            t = (px_x - xs[i]) / (xs[i + 1] - xs[i])
            return int(round(bs[i] + t * (bs[i + 1] - bs[i])))
    return bs[-1]


def draw(view, det, pending, points, posture, step, pose, msg):
    h, w = view.shape[:2]
    for p in points:                       # 기록된 점을 세로선으로
        x = p["px_x"]
        cv2.line(view, (x, 0), (x, h), (255, 200, 0), 1)
        cv2.putText(view, f"{p['base']}", (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 200, 0), 1)
    if det:
        bx, by, bw, bh = det["box"]
        cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.circle(view, (det["cx"], det["cy"]), 5, (0, 0, 255), -1)
        if len(points) >= 2:
            pred = predict(points, det["cx"])
            cv2.putText(view, f"예측 베이스 {pred}도", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(view, f"검출 x={det['cx']} 넓이 {int(det['area'])}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(view, "물체 없음", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if pending is not None:
        cv2.line(view, (pending, 0), (pending, h), (255, 0, 255), 2)

    panel = view.copy()
    cv2.rectangle(panel, (0, h - 96), (w, h), (0, 0, 0), -1)
    view[:] = cv2.addWeighted(panel, 0.62, view, 0.38, 0)

    if not posture:
        # 1단계: 아직 '가리키는 자세'를 안 정했다 — 자세 조정 키를 보여준다
        lines = [
            f"[1단계] 팔을 앞으로 뻗어 '가리키는 자세'를 만드세요  (관절 {pose[:4]}, 단위 {step}도)",
            "w/s 어깨   q/e 팔꿈치   z/c 손목   a/d 베이스   [ / ] 단위   h 홈",
            "자세가 잡히면  →  t  를 눌러 고정",
            msg,
        ]
    else:
        # 2단계: 자세는 고정. 이제 베이스만 돌린다
        lines = [
            f"[2단계] 자세 고정됨 {posture} | 베이스 {pose[0]}도 | 단위 {step}도 | 기록 {len(points)}점",
            "r 물체위치 기억  →  a/d 로 물체를 가리키기  →  space 저장",
            "u 취소   [ / ] 단위   h 홈   t 자세 다시 정하기   x 저장종료   ESC 취소종료",
            msg,
        ]
    for i, t in enumerate(lines):
        last = (i == len(lines) - 1)
        col = (0, 200, 255) if (last and msg.startswith("⚠")) else \
              (150, 255, 150) if (i == 2 and not posture) else (255, 255, 255)
        cv2.putText(view, t, (8, h - 74 + i * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(vision.CONFIG_FILE):
        print("⚠ vision_config.json이 없습니다. vision_check.py로 색을 먼저 맞추세요.")
        return 1
    cfg = vision.load_config()

    cap = vision.open_camera(args.cam)
    if cap is None:
        print(f"⚠ {args.cam}번 카메라를 열 수 없습니다.")
        return 1

    import serial
    ser = serial.Serial(args.port, 115200, timeout=0.5)
    time.sleep(2.5)
    ser.reset_input_buffer()
    bridge.send_serial(ser, ["SPEED 6", "HOME"])
    print(f"[연결됨] {args.port}")
    print("먼저 팔을 '가리키는 자세'로 만들고 t를 누르세요 (w/s 어깨, q/e 팔꿈치).")

    points, posture = [], None
    if os.path.exists(args.out) and not args.fresh:
        d = json.load(open(args.out, encoding="utf-8"))
        points, posture = d.get("points", []), d.get("posture")
        print(f"기존 {len(points)}점, 자세 {posture} 이어받음 (--fresh로 새로 시작)")

    step, pending, msg = 5, None, "가리키는 자세를 만들고 t"
    pose = bridge.query_pose(ser) or [90, 90, 90, 90, 60]
    cv2.namedWindow("point calib")
    frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det, _ = vision.find_object(frame, cfg["range"], cfg["min_area"])
        view = frame.copy()
        draw(view, det, pending, points, posture, step, pose, msg)
        cv2.imshow("point calib", view)

        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        if k == 27:
            print("저장하지 않고 종료합니다.")
            cap.release(); ser.close(); cv2.destroyAllWindows(); return 0
        if k == ord("x"):
            break
        if k == ord("["):
            step = max(1, step - 1)
        if k == ord("]"):
            step = min(20, step + 1)
        if k == ord("h"):
            bridge.send_serial(ser, ["HOME"]); pose = bridge.query_pose(ser) or pose
        if k in JOG:
            j, sign = JOG[k]
            lo, hi = bridge.JOINT_LIMITS[j]
            bridge.send_serial(ser, [f"S {j} {max(lo, min(hi, pose[j] + sign * step))}"])
            pose = bridge.query_pose(ser) or pose
        if k == ord("t"):
            posture = [int(v) for v in pose[1:4]]     # 어깨·팔꿈치·손목
            msg = f"가리키는 자세 고정 {posture} — 이제 물체를 놓고 r"
            print(f"  [t] 자세 고정: {posture}")

        if k == ord("r"):
            if not posture:
                msg = "⚠ 먼저 t로 가리키는 자세를 고정하세요"
            elif not det:
                msg = "⚠ 물체가 안 보입니다. 팔을 치우고 다시 r"
            elif any(abs(det["cx"] - p["px_x"]) < MIN_GAP_PX for p in points):
                msg = "⚠ 기존 기록과 가로로 너무 가깝습니다 — 좌우로 더 옮기세요"
            else:
                pending = det["cx"]
                msg = f"화면 x={pending} 기억 → a/d로 물체를 가리키고 space"
                print(f"  [r] x={pending}")

        if k == ord(" "):
            if pending is None:
                msg = "⚠ 먼저 r로 물체 위치를 기억하세요"
            else:
                cur = bridge.query_pose(ser)
                if cur is None:
                    msg = "⚠ 팔 각도를 읽지 못했습니다"
                else:
                    cand = points + [{"px_x": pending, "base": int(cur[0])}]
                    ok_mono, why = check_monotonic(cand)
                    if not ok_mono:
                        msg = f"⚠ {why}"
                        print(f"  ✗ 기록 거부: {why}")
                        print("     화면 왼쪽↔오른쪽과 베이스 각도는 한 방향으로 대응해야 합니다.")
                    else:
                        points = cand
                        print(f"  ✔ {len(points)}점: x={pending} → 베이스 {int(cur[0])}도")
                        msg = f"{len(points)}점 기록. 물체를 좌우로 옮기고 r"
                        pending = None

        if k == ord("u") and points:
            print(f"  취소: {points.pop()}")
            msg = "마지막 기록 취소"

    if posture and points:
        json.dump({"frame_size": [frame.shape[1], frame.shape[0]],
                   "posture": posture, "points": points,
                   "vision": {"range": cfg["range"], "min_area": cfg["min_area"]}},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        xs = [p["px_x"] for p in points]
        print(f"\n{len(points)}점 저장 → {os.path.abspath(args.out)}")
        print("── 품질 요약 " + "─" * 38)
        print(f"  가로 커버리지 {min(xs)}~{max(xs)} "
              f"(화면의 {(max(xs)-min(xs))/frame.shape[1]*100:.0f}%, 60% 이상 권장)")
        print("  " + check_monotonic(points)[1])
        if len(points) < 5:
            print(f"  ⚠ {len(points)}점 — 5점 이상을 권합니다")
    else:
        print("\n자세나 점이 없어 저장하지 않았습니다.")

    bridge.send_serial(ser, ["HOME"])
    ser.close(); cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
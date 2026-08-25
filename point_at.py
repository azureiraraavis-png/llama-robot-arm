# -*- coding: utf-8 -*-
"""
point_at.py — 카메라로 물체를 보고 팔이 그쪽을 가리킨다

보정표(point_calib.json)의 "화면 x → 베이스 각도" 대응을 선형 보간해서 쓴다.
LLM은 여기 관여하지 않는다 — 위치를 재고 각도를 내는 일은 기하학이다.

실행
  py -3.12 point_at.py              # 따라 보기 (물체를 움직이면 팔이 따라온다)
  py -3.12 point_at.py --once       # 한 번만 가리키고 끝
  py -3.12 point_at.py --dry-run    # 팔 없이 예측 각도만 확인

멈추지 않고 떨리는 것을 막기 위해
  · 불감대(deadband): 목표가 3도 미만으로 바뀌면 움직이지 않는다
  · 최소 간격: 0.35초에 한 번만 명령을 보낸다
  · 평활화: 최근 몇 프레임의 중앙값을 쓴다 (한 프레임 튐 무시)

  q 또는 ESC 로 종료
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2

import llm_arm_bridge as bridge
import vision
from calibrate_point import predict

CALIB = "point_calib.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--once", action="store_true", help="한 번만 가리키고 종료")
    ap.add_argument("--dry-run", action="store_true", help="팔 없이 예측만")
    ap.add_argument("--deadband", type=int, default=3, help="이 각도 미만 변화는 무시")
    ap.add_argument("--interval", type=float, default=0.35, help="명령 최소 간격(초)")
    args = ap.parse_args()

    if not os.path.exists(args.calib):
        print(f"⚠ {args.calib}이 없습니다. calibrate_point.py로 보정을 먼저 하세요.")
        return 1
    cal = json.load(open(args.calib, encoding="utf-8"))
    points, posture = cal["points"], cal["posture"]
    vcfg = cal.get("vision") or vision.load_config()
    rng, min_area = vcfg["range"], vcfg.get("min_area", 300)
    xs = [p["px_x"] for p in points]
    print(f"보정 {len(points)}점 (화면 x {min(xs)}~{max(xs)}), 가리키는 자세 {posture}")

    cap = vision.open_camera(args.cam)
    if cap is None:
        print(f"⚠ {args.cam}번 카메라를 열 수 없습니다.")
        return 1

    ser = None
    if not args.dry_run:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.5)
        time.sleep(2.5)
        ser.reset_input_buffer()
        print(f"[연결됨] {args.port}")
        bridge.send_serial(ser, ["SPEED 7"] +
                           [f"S {j} {a}" for j, a in zip((1, 2, 3), posture)])

    recent, last_sent, last_time = [], None, 0.0
    cv2.namedWindow("point at")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det, _ = vision.find_object(frame, rng, min_area)
        view = frame.copy()
        h, w = view.shape[:2]
        for p in points:
            cv2.line(view, (p["px_x"], h - 26), (p["px_x"], h), (255, 200, 0), 1)

        target = None
        if det:
            recent.append(det["cx"])
            recent = recent[-5:]
            smooth = int(statistics.median(recent))     # 한 프레임 튐 무시
            target = predict(points, smooth)
            outside = smooth < min(xs) or smooth > max(xs)
            bx, by, bw, bh = det["box"]
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.line(view, (smooth, 0), (smooth, h), (0, 255, 255), 1)
            cv2.putText(view, f"x={smooth} → 베이스 {target}도" +
                        ("  (보정 범위 밖 — 끝값 사용)" if outside else ""),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255) if outside else (0, 255, 255), 2)
        else:
            recent.clear()
            cv2.putText(view, "물체 없음", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)

        now = time.time()
        if target is not None and (last_sent is None or abs(target - last_sent) >= args.deadband) \
           and now - last_time >= args.interval:
            lo, hi = bridge.JOINT_LIMITS[0]
            t = max(lo, min(hi, target))
            if ser:
                bridge.send_serial(ser, [f"S 0 {t}"])
            else:
                print(f"[dry-run] S 0 {t}")
            last_sent, last_time = t, now
            if args.once:
                print(f"가리켰습니다: 베이스 {t}도")
                break

        cv2.putText(view, f"현재 지시 {last_sent if last_sent is not None else '-'}도  (q 종료)",
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("point at", view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break

    if ser:
        bridge.send_serial(ser, ["HOME"])
        ser.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
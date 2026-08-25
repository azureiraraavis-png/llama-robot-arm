# -*- coding: utf-8 -*-
"""
calibrate_arm.py — 2단계: "화면 위치 ↔ 관절 각도" 대응표를 실측으로 만든다

왜 역기구학(수학)을 안 쓰는가
  이 팔은 레이저 커팅 부품에 마이크로 서보라 링크 길이도, 서보 영점도,
  기어 유격도 정밀하지 않다. 계산으로 뽑은 각도는 실제와 어긋난다.
  그리퍼 닫힘 각도를 실측으로 120도로 정했듯이, 여기서도 실측이 답이다.

왜 카메라 위치를 몰라도 되는가
  픽셀 → 각도 대응을 직접 배우기 때문에 카메라의 위치·각도·렌즈가
  변환식에 들어가지 않는다. 단, 보정 후에 카메라가 움직이면 전부 무효다.

한 지점을 두 박자로 기록한다 (팔이 물체를 가리는 문제 때문)
  ① 팔을 치운 상태에서 물체 위치를 찍는다        [r]
  ② 팔을 몰고 가 집을 자세를 만든 뒤 각도를 찍는다 [스페이스]

준비: py -3.12 -m pip install opencv-python pyserial
실행: py -3.12 calibrate_arm.py            (COM4, 0번 카메라)
      py -3.12 calibrate_arm.py --cam 1 --port COM5

조작
  r        지금 보이는 물체 위치를 기억 (팔을 화면 밖으로 치우고 누를 것)
  w/s      어깨(1) 앞뒤        a/d  베이스(0) 좌우
  q/e      팔꿈치(2)           z/c  손목(3)
  o/p      그리퍼 열기/닫기
  [ / ]    조작 단위 감소/증가 (기본 5도)
  스페이스  현재 관절 각도를 기억한 물체 위치와 짝지어 저장
  u        마지막 기록 취소     h  홈 자세
  x        저장하고 종료        ESC 저장 없이 종료
"""

import argparse
import json
import os
import sys
import time

import cv2

import llm_arm_bridge as bridge
import vision

CALIB_FILE = "arm_calib.json"
JOG_JOINTS = {ord("a"): (0, -1), ord("d"): (0, +1),
              ord("s"): (1, -1), ord("w"): (1, +1),
              ord("q"): (2, -1), ord("e"): (2, +1),
              ord("z"): (3, -1), ord("c"): (3, +1)}


def draw_help(img, step, pending, n_points, pose):
    h, w = img.shape[:2]
    panel = img.copy()
    cv2.rectangle(panel, (0, h - 92), (w, h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(panel, 0.55, img, 0.45, 0)
    lines = [
        "r=물체위치 기억  space=각도 저장  u=취소  h=홈  x=저장종료",
        "a/d 베이스  w/s 어깨  q/e 팔꿈치  z/c 손목  o/p 그리퍼  [/] 단위",
        f"단위 {step}도 | 기록 {n_points}점 | " +
        (f"기억된 물체 위치 {pending}" if pending else "물체 위치 미기억 (r 먼저)"),
    ]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, h - 66 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)
    if pose:
        cv2.putText(img, f"관절 {pose[:4]}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--out", default=CALIB_FILE)
    args = ap.parse_args()

    cfg = vision.load_config()
    if not os.path.exists(vision.CONFIG_FILE):
        print("⚠ vision_config.json이 없습니다. vision_check.py로 색을 먼저 맞추세요.")
        return 1

    cap = vision.open_camera(args.cam)
    if cap is None:
        print(f"⚠ {args.cam}번 카메라를 열 수 없습니다.")
        return 1

    import serial
    ser = serial.Serial(args.port, 115200, timeout=0.5)
    time.sleep(2.5)
    ser.reset_input_buffer()
    print(f"[연결됨] {args.port}")
    bridge.send_serial(ser, ["SPEED 6", "HOME"])

    points = []
    if os.path.exists(args.out):           # 이어서 보정할 수 있게
        points = json.load(open(args.out, encoding="utf-8"))["points"]
        print(f"기존 보정점 {len(points)}개를 이어받았습니다.")

    step = 5
    pending = None                          # 기억해 둔 물체 픽셀 좌표
    pose = bridge.query_pose(ser) or [90, 90, 90, 90, 60]
    cv2.namedWindow("calibrate")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det, _ = vision.find_object(frame, cfg["range"], cfg["min_area"])
        view = frame.copy()

        if det:
            x, y, bw, bh = det["box"]
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(view, (det["cx"], det["cy"]), 5, (0, 0, 255), -1)
        if pending:
            cv2.drawMarker(view, tuple(pending), (255, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 26, 2)
        for p in points:                     # 이미 기록한 지점들
            cv2.circle(view, tuple(p["px"]), 7, (255, 200, 0), 2)

        draw_help(view, step, pending, len(points), pose)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue

        if k == 27:                                  # ESC
            print("저장하지 않고 종료합니다.")
            break
        if k == ord("x"):
            break
        if k == ord("["):
            step = max(1, step - 1)
        if k == ord("]"):
            step = min(20, step + 1)
        if k == ord("h"):
            bridge.send_serial(ser, ["HOME"])
            pose = bridge.query_pose(ser) or pose
        if k == ord("r"):
            if det:
                pending = [det["cx"], det["cy"]]
                print(f"  물체 위치 기억: {pending}  → 이제 팔을 몰고 가세요")
            else:
                print("  ⚠ 물체가 안 보입니다. 팔을 치우고 다시 r")
        if k in (ord("o"), ord("p")):
            bridge.send_serial(ser, [f"GRIP {0 if k == ord('o') else 1}"])
            pose = bridge.query_pose(ser) or pose
        if k in JOG_JOINTS:
            j, sign = JOG_JOINTS[k]
            lo, hi = bridge.JOINT_LIMITS[j]
            target = max(lo, min(hi, pose[j] + sign * step))
            bridge.send_serial(ser, [f"S {j} {target}"])
            pose = bridge.query_pose(ser) or pose
        if k == ord(" "):
            if not pending:
                print("  ⚠ 먼저 r로 물체 위치를 기억하세요")
            else:
                cur = bridge.query_pose(ser)
                if cur is None:
                    print("  ⚠ 팔 각도를 읽지 못했습니다")
                else:
                    points.append({"px": pending, "pose": [int(v) for v in cur[:4]]})
                    print(f"  ✔ {len(points)}번째 기록: 화면{pending} → 각도{cur[:4]}")
                    pending = None
        if k == ord("u") and points:
            gone = points.pop()
            print(f"  취소: {gone}")

    if k != 27:
        h, w = frame.shape[:2]
        data = {"frame_size": [w, h], "points": points,
                "vision": {"range": cfg["range"], "min_area": cfg["min_area"]}}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n보정점 {len(points)}개 저장 → {os.path.abspath(args.out)}")
        if len(points) < 6:
            print("※ 6점 이상이면 보간이 안정적입니다. 작업 공간 골고루 채워주세요.")

    bridge.send_serial(ser, ["HOME"])
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
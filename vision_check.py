# -*- coding: utf-8 -*-
"""
vision_check.py — 1단계: 카메라가 물체를 안정적으로 잡는지 확인하고 색 범위를 맞춘다

이 단계를 대충 넘기면 뒤가 전부 흔들린다. 팔을 붙이기 전에
"물체를 놓아둔 채 손을 흔들어도 중심이 튀지 않는" 상태를 만들어 두는 것이 목적이다.

준비:
  py -3.12 -m pip install opencv-python
  (headless 버전이 아니라 opencv-python 이어야 창이 뜬다)

실행:
  py -3.12 vision_check.py --list        # 연결된 카메라 번호 확인
  py -3.12 vision_check.py               # 0번 카메라
  py -3.12 vision_check.py --cam 1

조작
  1 2 3 4  빨강 / 초록 / 파랑 / 노랑 기본값 불러오기
  마우스   화면을 클릭하면 그 지점의 색으로 범위를 자동 설정
  트랙바   H/S/V 범위 미세 조정
  s        현재 설정을 vision_config.json 에 저장
  q        종료
"""

import argparse
import sys

import cv2
import numpy as np

import vision

WIN = "arm vision"
BARS = "range"
_click = {"pt": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click["pt"] = (x, y)


def build_bars(rng):
    cv2.namedWindow(BARS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(BARS, 420, 260)
    for name, val, mx in [("H min", rng["h"][0], 180), ("H max", rng["h"][1], 180),
                          ("S min", rng["s"][0], 255), ("S max", rng["s"][1], 255),
                          ("V min", rng["v"][0], 255), ("V max", rng["v"][1], 255),
                          ("wrap(빨강)", int(rng.get("wrap", False)), 1),
                          ("min area", 300, 5000)]:
        cv2.createTrackbar(name, BARS, val, mx, lambda v: None)


def read_bars():
    g = lambda n: cv2.getTrackbarPos(n, BARS)
    return ({"h": [g("H min"), g("H max")], "s": [g("S min"), g("S max")],
             "v": [g("V min"), g("V max")], "wrap": bool(g("wrap(빨강)"))},
            max(50, g("min area")))


def set_bars(rng):
    cv2.setTrackbarPos("H min", BARS, rng["h"][0]); cv2.setTrackbarPos("H max", BARS, rng["h"][1])
    cv2.setTrackbarPos("S min", BARS, rng["s"][0]); cv2.setTrackbarPos("S max", BARS, rng["s"][1])
    cv2.setTrackbarPos("V min", BARS, rng["v"][0]); cv2.setTrackbarPos("V max", BARS, rng["v"][1])
    cv2.setTrackbarPos("wrap(빨강)", BARS, int(rng.get("wrap", False)))


def range_from_pixel(frame, x, y, hw=12):
    """클릭한 지점 주변의 색을 보고 범위를 잡아준다 (미세조정 출발점)."""
    patch = frame[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (int(x) for x in np.median(hsv, axis=0))
    wrap = h < hw or h > 180 - hw          # 색상환 0도 부근이면 빨강 처리 필요
    return {"h": [max(0, h - hw), min(180, h + hw)],
            "s": [max(30, s - 70), 255], "v": [max(30, v - 70), 255], "wrap": bool(wrap)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="연결된 카메라 번호 조회")
    ap.add_argument("--diag", action="store_true", help="번호×백엔드 전부 시험 (문제 추적용)")
    ap.add_argument("--probe", action="store_true",
                    help="'열림+읽기 실패'일 때 해상도·포맷을 바꿔가며 재시도")
    args = ap.parse_args()

    if args.probe:
        print(f"OpenCV {cv2.__version__} — {args.cam}번 카메라 조합 탐색 (조합당 최대 5초)\n")
        rows = vision.probe_variants(args.cam)
        for be, label, ok, shape in rows:
            print(f"  {be:<11}{label:<14}{'✓ 프레임 나옴 ' + shape if ok else '✗'}")
        if rows and rows[-1][2]:
            be, label = rows[-1][0], rows[-1][1]
            print(f"\n성공 조합: {be} / {label}")
            print("이 조합을 기본으로 쓰도록 알려주세요 — 코드에 고정해 드리겠습니다.")
        else:
            print("\n모든 조합 실패. 소프트웨어 밖 문제일 가능성이 큽니다:")
            print("  · 설정 → 개인 정보 및 보안 → 카메라 →")
            print("    '카메라 액세스' 및 '데스크톱 앱이 카메라에 액세스하도록 허용' 켜기")
            print("  · 윈도우 기본 '카메라' 앱에서 영상이 나오는지 확인 (확인 후 반드시 닫기)")
            print("  · 다른 프로그램(줌·팀즈·브라우저 탭)이 점유 중인지")
            print("  · USB 포트를 바꿔 다시 연결")
        return 0

    if args.diag:
        print(f"OpenCV {cv2.__version__}")
        print(f"{'번호':<5}{'백엔드':<12}{'열기':<8}{'읽기'}")
        print("-" * 44)
        for i, be, opened, read in vision.diagnose():
            print(f"{i:<5}{be:<12}{opened:<8}{read}")
        print("\n하나라도 '읽기 OK'가 있으면 그 번호로 실행하세요: --cam <번호>")
        print("전부 '안 열림'이면 카메라가 윈도우에 인식되지 않은 것입니다:")
        print("  · 장치 관리자 → 카메라 항목에 장치가 보이는지")
        print("  · 설정 → 개인 정보 및 보안 → 카메라 → '데스크톱 앱이 액세스하도록 허용' 켜기")
        print("  · 줌/팀즈/카메라 앱 등 다른 프로그램이 쓰고 있지 않은지")
        return 0

    if args.list:
        found = vision.list_cameras()
        print("사용 가능한 카메라 번호:", found if found else "없음")
        if not found:
            print("→ 원인을 보려면: py -3.12 vision_check.py --diag")
        else:
            print("여러 개면 --cam 번호 로 골라 실행하세요.")
        return 0

    cap = vision.open_camera(args.cam)
    if cap is None:
        print(f"⚠ {args.cam}번 카메라를 열 수 없습니다. --list 로 번호를 확인하세요.")
        return 1

    cfg = vision.load_config()
    rng = cfg["range"]
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    build_bars(rng)
    print("색이 잘 잡히도록 트랙바를 조정하거나 물체를 클릭하세요. s=저장, q=종료")

    stable = []
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠ 프레임을 읽지 못했습니다.")
            break

        if _click["pt"]:
            x, y = _click["pt"]
            _click["pt"] = None
            rng = range_from_pixel(frame, x, y)
            set_bars(rng)
            print(f"  클릭 지점 색으로 범위 설정: H{rng['h']} S{rng['s']} V{rng['v']} "
                  f"wrap={rng['wrap']}")

        rng, min_area = read_bars()
        det, mask = vision.find_object(frame, rng, min_area)

        view = frame.copy()
        h, w = frame.shape[:2]
        cv2.line(view, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
        cv2.line(view, (0, h // 2), (w, h // 2), (90, 90, 90), 1)

        if det:
            x, y, bw, bh = det["box"]
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(view, (det["cx"], det["cy"]), 5, (0, 0, 255), -1)
            nx, ny = vision.norm_position(det, frame.shape)
            cv2.putText(view, f"({det['cx']},{det['cy']})  x={nx:+.2f} y={ny:+.2f}  area={int(det['area'])}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            stable.append((det["cx"], det["cy"]))
            stable = stable[-30:]
            if len(stable) == 30:
                arr = np.array(stable)
                jitter = float(np.max(arr, axis=0).max() - np.min(arr, axis=0).min())
                col = (0, 255, 0) if jitter < 8 else (0, 165, 255) if jitter < 20 else (0, 0, 255)
                cv2.putText(view, f"흔들림 {jitter:.0f}px (10 미만이면 좋음)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        else:
            stable.clear()
            cv2.putText(view, "물체 없음 - 색 범위를 조정하거나 물체를 클릭하세요",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow(WIN, view)
        cv2.imshow("mask", cv2.resize(mask, (w // 2, h // 2)))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in map(ord, "1234"):
            name = ["red", "green", "blue", "yellow"]["1234".index(chr(k))]
            rng = dict(vision.DEFAULT_RANGES[name])
            set_bars(rng)
            print(f"  기본값 불러옴: {name}")
        if k == ord("s"):
            rng, min_area = read_bars()
            path = vision.save_config({"color": "custom", "range": rng, "min_area": min_area})
            print(f"  저장됨 → {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
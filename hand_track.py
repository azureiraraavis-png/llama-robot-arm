# -*- coding: utf-8 -*-
"""
hand_track.py — 카메라로 손을 보고, 로봇팔이 따라 움직인다

주의 1: 최신 mediapipe(0.10.30 이상)에는 인터넷 예제에 흔한 `mp.solutions.hands`가 없습니다.
        Tasks API + 별도 모델 파일(hand_landmarker.task)을 씁니다(자동으로 내려받음, 약 8MB).
주의 2: 그런데 그 최신판은 서명 없는 libmediapipe.dll을 불러오기 때문에,
        윈도우 '스마트 앱 제어'가 켜져 있으면 WinError 4551로 막힙니다.
        그럴 때는 0.10.21로 낮추면 됩니다(아래 참조). 이 프로그램은 두 방식을 모두 지원합니다.

준비
  py -3.12 -m pip install "mediapipe==0.10.21" opencv-python pyserial
  (스마트 앱 제어가 꺼져 있다면 그냥 mediapipe 최신판도 됩니다)

실행 (반드시 --dry-run 부터)
  py -3.12 hand_track.py --dry-run     팔 없이 인식만. 창을 보며 감을 잡는 단계
  py -3.12 hand_track.py               COM4, 0번 카메라, 실제로 팔이 움직임
  py -3.12 hand_track.py --no-grip     집게 제어 끄기 (그리퍼 선이 불안할 때)
  py -3.12 hand_track.py --gentle      더 느리고 작게 (처음엔 이걸 권합니다)

조작 (창을 클릭해 활성화한 뒤 키를 누르세요)
  스페이스  일시정지 / 재개   — 팔이 그 자세로 멈춥니다
  h  홈 자세      i  좌우 방향 뒤집기      g  집게 제어 켜기/끄기
  q 또는 ESC  종료 (팔은 홈으로 돌아갑니다)

손 → 팔
  좌우 위치 → 베이스     상하 위치 → 어깨      손 크기(카메라와의 거리) → 팔꿈치
  손 기울기 → 손목       엄지·검지 벌림 → 집게

화면 글자는 영어입니다. OpenCV 기본 글꼴이 한글을 그리지 못해 ???로 나오기 때문입니다.
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2

import hand_map

MODEL_FILE = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# 손 골격 연결. mediapipe의 그리기 도구도 solutions와 함께 사라져서 직접 그립니다.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "grip"]


# ── 모델 ────────────────────────────────────────────────────────────────

def ensure_model(path=MODEL_FILE):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    print(f"손 인식 모델을 내려받습니다 (약 8MB)\n  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print(f"  완료 → {os.path.abspath(path)}")
        return path
    except Exception as e:
        print(f"⚠ 내려받기 실패: {e}")
        print("  브라우저로 위 주소를 열어 파일을 받은 뒤,")
        print(f"  이 스크립트와 같은 폴더에 '{MODEL_FILE}' 이름으로 두고 다시 실행하세요.")
        return None


# ── 손 인식기: 두 가지 방식 ─────────────────────────────────────────────
# mediapipe는 버전에 따라 API가 다릅니다.
#   0.10.30 이상 : Tasks API만 있음. libmediapipe.dll을 ctypes로 불러오는데,
#                  이 DLL은 서명이 없어서 윈도우 '스마트 앱 제어'가 막습니다(WinError 4551).
#   0.10.21 이하 : Tasks API + 옛 solutions API 둘 다 있음. .pyd 확장모듈이라 대체로 통과됩니다.
# 그래서 둘 다 지원하고, 되는 쪽을 씁니다.

class TasksSource:
    """새 방식. hand_landmarker.task 모델 파일이 필요합니다."""
    name = "tasks"

    def __init__(self, model_path):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,                       # 손 하나만 본다
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        import mediapipe as mp
        self.mp = mp

    def detect(self, rgb, ts_ms):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, ts_ms)
        if not res.hand_landmarks:
            return None
        return [(p.x, p.y) for p in res.hand_landmarks[0]]

    def close(self):
        self.lm.close()


class SolutionsSource:
    """옛 방식. 모델이 패키지 안에 들어 있어 따로 내려받을 필요가 없습니다."""
    name = "solutions"

    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)

    def detect(self, rgb, ts_ms):
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        return [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]

    def close(self):
        self.hands.close()


def make_hand_source(prefer="auto"):
    """되는 인식기를 만들어 돌려줍니다. 둘 다 실패하면 None."""
    errors = []
    order = {"auto": ["tasks", "solutions"],
             "tasks": ["tasks"], "solutions": ["solutions"]}[prefer]
    for which in order:
        try:
            if which == "tasks":
                path = ensure_model()
                if path is None:
                    raise RuntimeError("모델 파일 없음")
                src = TasksSource(path)
            else:
                src = SolutionsSource()
            print(f"[손 인식] {which} 방식")
            return src
        except Exception as e:
            errors.append((which, e))
    print("⚠ 손 인식기를 만들지 못했습니다.")
    for which, e in errors:
        print(f"   {which}: {type(e).__name__}: {e}")
    if any("4551" in str(e) for _w, e in errors):
        print("\n  → 윈도우 '스마트 앱 제어'가 mediapipe의 서명 없는 DLL을 막고 있습니다.")
        print("     해결: py -3.12 -m pip install \"mediapipe==0.10.21\"")
    return None


# ── 시리얼 (조용한 판) ──────────────────────────────────────────────────
# llm_arm_bridge.send_serial은 한 줄마다 화면에 찍습니다. 초당 십여 번 보내는
# 이 프로그램에서는 콘솔이 넘쳐버리므로, 여기서는 조용히 보내고 OK만 기다립니다.
# OK를 기다리는 것 자체가 속도 제한 역할을 합니다 — 팔이 다 움직여야 다음이 갑니다.

def send_quiet(ser, line, timeout=2.0):
    ser.write((line + "\n").encode())
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = ser.readline().decode(errors="ignore").strip()
        if not resp:
            continue
        if resp == "OK":
            return True
        if resp.startswith("ERR"):
            print(f"  ⚠ 아두이노 오류: {resp}  ({line})")
            return True
    return False


def open_arm(port, speed):
    import serial
    ser = serial.Serial(port, 115200, timeout=0.3)
    time.sleep(2.5)                       # 아두이노 리셋 대기
    ser.reset_input_buffer()
    send_quiet(ser, f"SPEED {max(1, min(10, speed))}")
    send_quiet(ser, "HOME", timeout=6.0)
    print(f"[연결됨] {port}")
    return ser


# ── 카메라 ──────────────────────────────────────────────────────────────

def open_cam(index):
    try:
        import vision as camera_util          # 이전에 만든 카메라 유틸이 있으면 재사용
        cap = camera_util.open_camera(index)
        if cap is not None:
            return cap
    except Exception:
        pass
    for be in (getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), cv2.CAP_ANY):
        if be is None:
            continue
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            for _ in range(10):               # 첫 프레임이 늦게 나오는 카메라 대비
                ok, _f = cap.read()
                if ok:
                    return cap
                time.sleep(0.2)
        cap.release()
    return None


# ── 그리기 ──────────────────────────────────────────────────────────────

def draw_hand(view, lm_px):
    for a, b in CONNECTIONS:
        cv2.line(view, lm_px[a], lm_px[b], (80, 220, 80), 2)
    for i, p in enumerate(lm_px):
        big = i in (0, 4, 5, 8, 17)
        cv2.circle(view, p, 6 if big else 3,
                   (0, 0, 255) if big else (200, 200, 0), -1)
    cv2.line(view, lm_px[4], lm_px[8], (255, 0, 255), 2)      # 핀치(집게) 선


def draw_bars(view, pose, x0, y0):
    """관절 5개를 막대로. 숫자보다 눈에 빨리 들어옵니다."""
    for i, (v, (lo, hi)) in enumerate(zip(pose, hand_map.SAFE)):
        y = y0 + i * 18
        t = (v - lo) / float(hi - lo) if hi > lo else 0.5
        cv2.rectangle(view, (x0 + 62, y - 9), (x0 + 62 + 120, y + 3), (60, 60, 60), -1)
        cv2.rectangle(view, (x0 + 62, y - 9), (x0 + 62 + int(120 * t), y + 3),
                      (0, 200, 255), -1)
        cv2.putText(view, f"{JOINT_NAMES[i]:<8}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)
        cv2.putText(view, str(v), (x0 + 190, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)


# ── 본체 ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="손을 따라 움직이는 로봇팔")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--dry-run", action="store_true", help="팔 없이 인식만")
    ap.add_argument("--no-grip", action="store_true", help="집게 제어 끄기")
    ap.add_argument("--invert", action="store_true", help="좌우 방향이 반대로 느껴질 때")
    ap.add_argument("--gentle", action="store_true", help="더 느리고 작게 움직이기")
    ap.add_argument("--speed", type=int, default=8, help="서보 속도 1~10 (기본 8)")
    ap.add_argument("--min-interval", type=float, default=0.07,
                    help="전송 최소 간격(초). 시리얼이 밀릴 때 늘리세요")
    ap.add_argument("--api", choices=["auto", "tasks", "solutions"], default="auto",
                    help="손 인식 방식 (기본 auto: 되는 쪽을 씀)")
    args = ap.parse_args()

    try:
        import mediapipe as mp          # noqa: F401  (버전 확인용)
        print(f"[mediapipe] {mp.__version__}")
    except ImportError:
        print("⚠ mediapipe가 없습니다:  py -3.12 -m pip install mediapipe opencv-python")
        return 1

    landmarker = make_hand_source(args.api)
    if landmarker is None:
        return 1

    cap = open_cam(args.cam)
    if cap is None:
        print(f"⚠ {args.cam}번 카메라를 열 수 없습니다.")
        print("  윈도우 '카메라' 앱이나 화상회의 프로그램이 켜져 있으면 닫아 주세요.")
        landmarker.close()
        return 1
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # 지연 줄이기
    except Exception:
        pass

    ser = None
    if not args.dry_run:
        try:
            ser = open_arm(args.port, args.speed)
        except Exception as e:
            print(f"⚠ {args.port} 연결 실패: {e}")
            print("  아두이노 IDE의 시리얼 모니터가 열려 있으면 닫고 다시 시도하세요.")
            print("  (팔 없이 보려면 --dry-run)")
            cap.release()
            landmarker.close()
            return 1

    sm = hand_map.Smoother(
        alpha=0.25 if args.gentle else 0.35,
        deadband=4 if args.gentle else 3,
        max_step=6 if args.gentle else 10,
    )
    invert = args.invert
    use_grip = not args.no_grip
    paused = False
    sent_n = 0
    last_send = 0.0
    fps, fps_t, fps_n = 0.0, time.time(), 0
    stamp = 0                 # mediapipe VIDEO 모드는 타임스탬프가 반드시 증가해야 합니다
    t0 = time.time()

    print("\n손을 카메라에 보여주세요.")
    print("  스페이스=일시정지  h=홈  i=좌우반전  g=집게  q=종료\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠ 카메라에서 프레임을 읽지 못했습니다.")
            break

        frame = cv2.flip(frame, 1)              # 거울처럼 보이게
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stamp = max(stamp + 1, int((time.time() - t0) * 1000))
        try:
            lm = landmarker.detect(rgb, stamp)
        except Exception as e:
            print(f"⚠ 인식 오류: {e}")
            break

        view = frame
        target = None
        if lm:
            lm_px = [(int(x * w), int(y * h)) for x, y in lm]
            draw_hand(view, lm_px)
            target = hand_map.hand_to_pose(lm, mirror=invert)
            if target is not None and not use_grip:
                target[4] = sm.sent[4]          # 집게는 지금 값 그대로
        else:
            cv2.putText(view, "no hand - holding pose", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not paused:
            now = time.time()
            if now - last_send >= args.min_interval:
                nxt = sm.update(target)
                if nxt is not None:
                    last_send = now
                    sent_n += 1
                    if ser and not send_quiet(ser, "P " + " ".join(str(v) for v in nxt)):
                        print("⚠ 팔이 응답하지 않습니다 — 연결과 전원을 확인하세요.")
                        break

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

        draw_bars(view, sm.sent, 10, h - 128)
        bar = (f"{'PAUSED' if paused else 'FOLLOW'} | grip {'on' if use_grip else 'off'}"
               f" | dir {'inv' if invert else 'nor'} | sent {sent_n} | {fps:.0f}fps"
               f"{' | DRY' if ser is None else ''}")
        cv2.rectangle(view, (0, h - 24), (w, h), (0, 0, 0), -1)
        cv2.putText(view, bar, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(view, "space=pause  h=home  i=invert  g=grip  q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow("hand track", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
            print("  일시정지" if paused else "  재개")
        elif k == ord("h"):
            if ser:
                send_quiet(ser, "HOME", timeout=6.0)
            sm.reset_to(hand_map.HOME)
            print("  홈")
        elif k == ord("i"):
            invert = not invert
            print(f"  좌우 방향 {'반전' if invert else '기본'}")
        elif k == ord("g"):
            use_grip = not use_grip
            print(f"  집게 제어 {'켬' if use_grip else '끔'}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        send_quiet(ser, "SPEED 5")
        send_quiet(ser, "HOME", timeout=6.0)
        ser.close()
    print(f"\n종료. 전송 {sent_n}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
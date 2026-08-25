# -*- coding: utf-8 -*-
"""
vision.py — 카메라에서 물체 위치를 찾는 부분 (팔과 무관한 순수 시각 처리)

역할 분담을 지킨다:
  · 이 파일은 "화면 어디에 무엇이 있는가"만 답한다 (픽셀 좌표)
  · 팔을 어떻게 움직일지는 이 파일이 모른다 (다음 단계에서 보정표가 담당)
  · LLM도 여기 관여하지 않는다 — 위치를 재는 일은 기하학이지 언어가 아니다

색으로 물체를 찾는다. 형태 인식보다 단순하지만, 조명만 일정하면
초당 수십 번 안정적으로 잡히고 원인 파악도 쉽다.
"""

import json
import os

import cv2
import numpy as np

CONFIG_FILE = "vision_config.json"

# 처음 시작할 때 쓸 기본 색 범위 (HSV). 조명에 따라 반드시 조정해야 한다.
DEFAULT_RANGES = {
    "red":    {"h": [0, 10],    "s": [120, 255], "v": [80, 255], "wrap": True},
    "green":  {"h": [40, 85],   "s": [80, 255],  "v": [60, 255], "wrap": False},
    "blue":   {"h": [95, 130],  "s": [90, 255],  "v": [60, 255], "wrap": False},
    "yellow": {"h": [20, 35],   "s": [110, 255], "v": [110, 255], "wrap": False},
}


def make_mask(frame_bgr, rng):
    """HSV 범위로 이진 마스크를 만든다.

    빨강은 색상환의 0도 부근이라 0~10과 170~180 두 구간으로 갈린다.
    wrap=True면 반대쪽 끝 구간도 함께 잡는다 (이걸 빼먹으면 빨강이 절반만 잡힌다).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # 범위 값은 JSON·트랙바·클릭 등 여러 곳에서 온다. numpy 정수가 섞이면
    # cv2.inRange가 거부하므로 여기서 순수 int로 통일한다.
    h0, h1 = int(rng["h"][0]), int(rng["h"][1])
    s0, s1 = int(rng["s"][0]), int(rng["s"][1])
    v0, v1 = int(rng["v"][0]), int(rng["v"][1])
    mask = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    if rng.get("wrap"):
        span = h1 - h0
        mask2 = cv2.inRange(hsv, (max(0, 180 - span), s0, v0), (180, s1, v1))
        mask = cv2.bitwise_or(mask, mask2)

    # 잡티 제거 → 구멍 메우기
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _contours(mask):
    """findContours의 반환 개수가 OpenCV 버전마다 달라서(2개 또는 3개) 흡수한다."""
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[-2] if len(res) == 3 else res[0]


def find_object(frame_bgr, rng, min_area=300):
    """가장 큰 색 덩어리의 중심을 찾는다.

    반환: dict(cx, cy, area, box, mask) 또는 None
      cx, cy : 픽셀 중심 (정수)
      area   : 넓이 (px²) — 거리 가늠에 쓸 수 있다
      box    : (x, y, w, h) 외접 사각형
    """
    mask = make_mask(frame_bgr, rng)
    cnts = _contours(mask)
    if len(cnts) == 0:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None, mask
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return {"cx": cx, "cy": cy, "area": float(area),
            "box": cv2.boundingRect(c)}, mask


def norm_position(det, frame_shape):
    """픽셀 좌표를 화면 비율(-1.0 ~ +1.0)로 바꾼다.
    해상도가 달라져도 같은 뜻이 되도록. x는 왼쪽이 -1, y는 위쪽이 -1."""
    h, w = frame_shape[:2]
    return ((det["cx"] - w / 2) / (w / 2), (det["cy"] - h / 2) / (h / 2))


def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"color": "red", "range": dict(DEFAULT_RANGES["red"]), "min_area": 300}


def save_config(cfg, path=CONFIG_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def _backends():
    """쓸 수 있는 카메라 백엔드 목록. 버전에 따라 없는 상수도 있어 확인 후 담는다."""
    out = []
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):   # 윈도우에선 DSHOW가 대체로 빠르다
        if hasattr(cv2, name):
            out.append(getattr(cv2, name))
    return out or [0]


def warmup_read(cap, tries=12, delay=0.12):
    """웹캠은 열린 직후 첫 프레임이 비는 경우가 흔하다. 잠깐 기다리며 재시도한다."""
    import time
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return True
        time.sleep(delay)
    return False


def open_camera(index=0, backend=None):
    backends = [backend] if backend is not None else _backends()
    for be in backends:
        try:
            cap = cv2.VideoCapture(index, be)
        except Exception:
            continue
        if cap.isOpened() and warmup_read(cap):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    return None


def backend_name(be):
    for n in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY", "CAP_V4L2"):
        if hasattr(cv2, n) and getattr(cv2, n) == be:
            return n
    return str(be)


def probe_variants(index=0, warm_seconds=5.0):
    """'열림 + 읽기 실패'일 때, 설정을 바꿔가며 프레임이 나오는 조합을 찾는다.

    카메라마다 받아들이는 해상도·포맷이 달라서, 기본값으로는 열려도
    프레임이 안 나오는 경우가 있다. 조합을 바꿔가며 시험한다.
    """
    import time
    variants = [
        ("기본값", None, None),
        ("MJPG", "MJPG", None),
        ("MJPG 640x480", "MJPG", (640, 480)),
        ("YUY2 640x480", "YUY2", (640, 480)),
        ("640x480", None, (640, 480)),
        ("320x240", None, (320, 240)),
        ("1280x720", None, (1280, 720)),
    ]
    results = []
    for be in _backends():
        for label, fourcc, size in variants:
            try:
                cap = cv2.VideoCapture(index, be)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            try:
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                if size:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            except Exception:
                pass
            deadline = time.time() + warm_seconds       # 넉넉히 기다려 본다
            ok, shape = False, ""
            while time.time() < deadline:
                r, f = cap.read()
                if r and f is not None and f.size:
                    ok, shape = True, f"{f.shape[1]}x{f.shape[0]}"
                    break
                time.sleep(0.15)
            cap.release()
            results.append((backend_name(be), label, ok, shape))
            if ok:
                return results   # 성공하면 거기서 멈춘다
    return results


def diagnose(max_index=4):
    """번호×백엔드 조합을 전부 시험해 어디서 걸리는지 보여준다."""
    rows = []
    for i in range(max_index):
        for be in _backends():
            try:
                cap = cv2.VideoCapture(i, be)
            except Exception as e:
                rows.append((i, backend_name(be), "예외", type(e).__name__))
                continue
            opened = cap.isOpened()
            read_ok = warmup_read(cap, tries=6) if opened else False
            size = ""
            if read_ok:
                ok, f = cap.read()
                if ok and f is not None:
                    size = f"{f.shape[1]}x{f.shape[0]}"
            cap.release()
            rows.append((i, backend_name(be),
                         "열림" if opened else "안 열림",
                         (f"읽기 OK {size}" if read_ok else "읽기 실패") if opened else "-"))
    return rows


def list_cameras(max_index=4):
    """연결된 카메라 번호를 훑어본다. 없는 번호에서 몇 초씩 걸릴 수 있다."""
    found = []
    for i in range(max_index):
        print(f"  {i}번 확인 중...", flush=True)
        cap = open_camera(i)
        if cap is not None:
            found.append(i)
            cap.release()
    return found
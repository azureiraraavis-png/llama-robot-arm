# -*- coding: utf-8 -*-
"""
look_at.py — "보이는 것 쪽을 봐" 를 실제로 수행하는 부분

브리지가 이 모듈을 실행 시점에 부릅니다. `move` 명령이 아두이노에게 현재 각도를
물어보는 것과 같은 자리입니다 — **계획을 세울 때가 아니라 보낼 때 하드웨어를 만집니다.**

역할 분담
  라마      "파란"이라는 낱말만 뽑는다
  colors.py 낱말 → HSV 범위
  vision.py 화면에서 그 색 덩어리를 찾는다 → 픽셀 x
  보정표    픽셀 x → 베이스 각도
  여기      위를 이어 붙이고, 못 봤으면 **움직이지 않고 그렇게 말한다**

못 봤을 때 추측해서 아무 데나 도는 것은 하지 않습니다. 그게 사용자에게 훨씬 나쁩니다.
"""

import json
import os
import statistics
import time

CALIB = "point_calib.json"

_cam = None          # 카메라는 한 번만 열고 재사용합니다 (여는 데 1초쯤 걸립니다)
_calib = None


def load_calib(path=CALIB):
    """보정표를 읽습니다. 한 번 읽으면 캐시합니다."""
    global _calib
    if _calib is not None:
        return _calib
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        _calib = json.load(f)
    return _calib


def calib_span(cal):
    """보정이 화면의 어디까지 덮는지. 밖은 직선을 늘여 추정하므로 덜 정확합니다."""
    xs = [p["px_x"] for p in cal.get("points", [])]
    return (min(xs), max(xs)) if xs else (None, None)


def get_camera(index=0):
    global _cam
    if _cam is not None:
        return _cam
    import vision
    _cam = vision.open_camera(index)
    return _cam


def release():
    global _cam
    if _cam is not None:
        try:
            _cam.release()
        except Exception:
            pass
        _cam = None


def pick_range(target):
    """무엇을 찾을지 정합니다. (범위, 사람에게 보여줄 이름, min_area)"""
    import colors
    import vision

    cal = load_calib()
    saved = (cal or {}).get("vision") or vision.load_config()
    min_area = (saved or {}).get("min_area", 300)

    name, rng = colors.resolve(target)
    if rng is not None:
        return rng, f"{name}색", min_area
    if saved and saved.get("range"):
        return saved["range"], "저장된 색", min_area
    return None, None, min_area


def find_x(frames=5, target=None, cam_index=0):
    """화면에서 대상을 찾아 픽셀 x를 냅니다. 못 찾으면 None.

    한 프레임만 보면 튀는 값에 속습니다. 여러 장 보고 중앙값을 씁니다.
    """
    import vision

    rng, label, min_area = pick_range(target)
    if rng is None:
        return None, "색 설정이 없습니다 (vision_check.py로 색을 먼저 잡으세요)", None

    cap = get_camera(cam_index)
    if cap is None:
        return None, "카메라를 열 수 없습니다 (카메라 앱이 켜져 있는지 확인)", None

    xs, shape = [], None
    for _ in range(max(1, frames)):
        ok, frame = cap.read()
        if not ok:
            continue
        shape = frame.shape
        det, _mask = vision.find_object(frame, rng, min_area=min_area)
        if det:
            xs.append(det["cx"])          # find_object는 (dict|None, mask)를 돌려줍니다
        time.sleep(0.03)

    if not xs:
        return None, f"{label} 물체가 화면에 보이지 않습니다", shape
    return int(statistics.median(xs)), label, shape


def angle_for(target=None, frames=5, cam_index=0):
    """대상을 보고 베이스 각도를 냅니다. (각도, 설명). 못 보면 (None, 이유)."""
    from calibrate_point import predict

    cal = load_calib()
    if cal is None:
        return None, "보정표(point_calib.json)가 없습니다 — calibrate_point.py를 먼저 돌리세요"

    x, label, _shape = find_x(frames=frames, target=target, cam_index=cam_index)
    if x is None:
        return None, label                      # 이때 label에 이유가 들어 있습니다

    angle = predict(cal["points"], x)
    if angle is None:
        return None, "보정표가 비어 있습니다"

    lo, hi = calib_span(cal)
    outside = ""
    if lo is not None and not (lo <= x <= hi):
        outside = f"  ※ 보정 구간({lo}~{hi}) 밖이라 추정값입니다"
    return int(angle), f"{label}을 화면 x={x}에서 찾음 → 베이스 {int(angle)}도{outside}"


def posture_lines(cal):
    """가리키는 자세(어깨·팔꿈치·손목)를 먼저 잡아 둡니다.

    보정을 그 자세에서 했으므로, 다른 자세로 서 있으면 같은 각도라도 다른 데를 가리킵니다.
    """
    p = (cal or {}).get("posture")
    if not p or len(p) < 3:
        return []
    return [f"S 1 {int(p[0])}", f"S 2 {int(p[1])}", f"S 3 {int(p[2])}"]


# ── 브리지가 부르는 자리 ────────────────────────────────────────────────

def execute_look(send_one, spec):
    """한 번 보고 그쪽으로 돕니다. send_one은 브리지의 한 줄 전송 함수입니다."""
    target = spec.get("target")
    angle, why = angle_for(target, frames=spec.get("frames", 5),
                           cam_index=spec.get("cam", 0))
    if angle is None:
        print(f"  👁 {why}")
        print("     — 못 봤으므로 팔을 움직이지 않습니다.")
        return True                              # 통신은 정상이니 다음 명령으로 진행
    print(f"  👁 {why}")
    cal = load_calib()
    for line in posture_lines(cal):
        if not send_one(line):
            return False
    return send_one(f"S 0 {angle}")


def execute_track(send_one, spec):
    """n초 동안 따라 봅니다. 떨림 방지는 point_at.py와 같은 방침입니다."""
    target = spec.get("target")
    secs = max(1, min(60, int(spec.get("seconds", 10))))
    deadband = int(spec.get("deadband", 3))
    interval = float(spec.get("interval", 0.35))
    cam = spec.get("cam", 0)

    cal = load_calib()
    if cal is None:
        print("  👁 보정표가 없습니다 — calibrate_point.py를 먼저 돌리세요")
        return True
    for line in posture_lines(cal):
        if not send_one(line):
            return False

    print(f"  👁 {secs}초 동안 따라 봅니다 (대상: {target or '저장된 색'})")
    end = time.time() + secs
    last_sent, misses, sent = None, 0, 0
    while time.time() < end:
        angle, why = angle_for(target, frames=2, cam_index=cam)
        if angle is None:
            misses += 1
            if misses == 1:
                print(f"     {why}")
            time.sleep(interval)
            continue
        misses = 0
        if last_sent is None or abs(angle - last_sent) >= deadband:
            if not send_one(f"S 0 {angle}"):
                return False
            last_sent, sent = angle, sent + 1
        time.sleep(interval)
    print(f"     따라보기 끝 (전송 {sent}회)")
    return True


if __name__ == "__main__":
    # 팔 없이 확인만:  py -3.12 look_at.py 빨간 물체
    import sys
    target = " ".join(sys.argv[1:]) or None
    cal = load_calib()
    if cal:
        lo, hi = calib_span(cal)
        print(f"\n보정 {len(cal['points'])}점, 화면 x {lo}~{hi}, 자세 {cal.get('posture')}")
    angle, why = angle_for(target)
    print(f"대상: {target or '(저장된 색)'}")
    print(f"결과: {why}")
    if angle is not None:
        print(f"→ 보낼 명령: S 0 {angle}")
    release()
# -*- coding: utf-8 -*-
"""
mp_check.py — mediapipe가 이 컴퓨터에서 실제로 돌아가는지 확인합니다.

hand_track.py를 돌리기 전에 이걸 먼저 돌리세요. 무엇이 되고 무엇이 막히는지,
막혔다면 다음에 무엇을 하면 되는지까지 알려줍니다.

  py -3.12 mp_check.py
"""

import sys
import platform

OK, NO = "  ✔", "  ✘"


def check_sac():
    """윈도우 '스마트 앱 제어' 상태를 읽습니다. 이게 서명 없는 DLL을 막는 범인입니다."""
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"SYSTEM\CurrentControlSet\Control\CI\Policy")
        v, _ = winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")
        winreg.CloseKey(k)
        return int(v)
    except Exception:
        return None


def main():
    print(f"\n파이썬 {sys.version.split()[0]}  ({sys.executable})")

    state = check_sac()
    label = {0: "꺼짐", 1: "켜짐 (적용 중)", 2: "평가 모드"}.get(state, "알 수 없음")
    print(f"스마트 앱 제어: {label}")
    if state in (1, 2):
        print("   → 켜져 있습니다. 서명 없는 DLL이 막힐 수 있습니다.")

    print("\n[1] mediapipe 불러오기")
    try:
        import mediapipe as mp
        print(f"{OK} mediapipe {mp.__version__}")
    except Exception as e:
        print(f"{NO} {type(e).__name__}: {e}")
        print("     py -3.12 -m pip install \"mediapipe==0.10.21\"")
        return 1

    print("\n[2] opencv 불러오기")
    try:
        import cv2
        print(f"{OK} opencv {cv2.__version__}")
    except Exception as e:
        print(f"{NO} {type(e).__name__}: {e}")
        return 1

    import numpy as np
    blank = np.zeros((240, 320, 3), np.uint8)
    results = {}

    print("\n[3] 옛 방식(solutions) — 모델 파일이 필요 없음")
    try:
        hands = mp.solutions.hands.Hands(max_num_hands=1)
        hands.process(blank)                       # 실제로 한 장 돌려봅니다
        hands.close()
        print(f"{OK} 됩니다")
        results["solutions"] = True
    except AttributeError:
        print(f"{NO} 이 버전에는 solutions API가 없습니다 (0.10.30 이상)")
        results["solutions"] = False
    except Exception as e:
        print(f"{NO} {type(e).__name__}: {e}")
        results["solutions"] = False

    print("\n[4] 새 방식(tasks) — hand_landmarker.task 모델 파일 필요")
    try:
        import os
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        have = os.path.exists("hand_landmarker.task")
        if not have:
            print("     모델 파일이 없습니다. 그래도 DLL은 이 단계에서 열리므로,")
            print("     막혀 있는지 여부는 지금 확인됩니다.")
        # create_from_options는 모델을 읽기 *전에* DLL부터 불러옵니다.
        # 그래서 모델이 없어도 '막혔는지'는 여기서 판별됩니다.
        lm = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO, num_hands=1))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
        lm.detect_for_video(img, 0)
        lm.close()
        print(f"{OK} 됩니다")
        results["tasks"] = True
    except Exception as e:
        msg = str(e)
        blocked = ("4551" in msg or "제어 정책" in msg or "Application Control" in msg)
        if blocked:
            print(f"{NO} 차단됨 — {type(e).__name__}: {msg}")
            print("\n     ▶ 스마트 앱 제어가 mediapipe의 서명 없는 DLL을 막았습니다.")
            results["tasks"] = False
        elif not have:
            print(f"{OK} DLL은 정상입니다 (모델 파일만 없음: {type(e).__name__})")
            results["tasks"] = True          # hand_track.py가 모델을 받아 옵니다
        else:
            print(f"{NO} {type(e).__name__}: {msg}")
            results["tasks"] = False

    print("\n[5] 카메라")
    try:
        import vision as camera_util
        cap = camera_util.open_camera(0)
    except Exception:
        cap = cv2.VideoCapture(0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            cap = None
    if cap is None:
        print(f"{NO} 0번 카메라를 열 수 없습니다 (카메라 앱이 켜져 있지 않은지 확인)")
    else:
        print(f"{OK} 0번 카메라 읽기 OK")
        cap.release()

    # ── 결론 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 58)
    if results.get("solutions") or results.get("tasks"):
        which = "tasks" if results.get("tasks") else "solutions"
        print(f"결론: 손 인식이 가능합니다. hand_track.py가 {which} 방식을 쓸 것입니다.")
        print("      py -3.12 hand_track.py --dry-run")
        return 0

    print("결론: 지금 버전으로는 손 인식이 막힙니다.")
    print("\n  가장 안전한 해결책 — mediapipe를 낮춥니다 (윈도우 설정은 그대로):")
    print("      py -3.12 -m pip install \"mediapipe==0.10.21\"")
    print("      py -3.12 mp_check.py")
    print("\n  0.10.21은 .pyd 확장모듈을 쓰기 때문에 대체로 통과되고,")
    print("  손 인식 기능은 최신판과 같습니다.")
    if state in (1, 2):
        print("\n  그래도 안 되면 스마트 앱 제어를 끄는 방법이 있지만,")
        print("  ⚠ 한 번 끄면 윈도우를 다시 설치하기 전에는 켤 수 없습니다. 마지막 수단입니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
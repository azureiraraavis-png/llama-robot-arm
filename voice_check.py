# -*- coding: utf-8 -*-
"""
voice_check.py — 음성 쪽이 이 컴퓨터에서 실제로 되는지 확인합니다.

voice_arm.py를 돌리기 전에 이걸 먼저 돌리세요. 마이크·GPU·받아쓰기를
따로따로 시험해서, 안 되는 게 있으면 어느 단계인지 바로 드러납니다.

  py -3.12 voice_check.py           마이크만 (빠름)
  py -3.12 voice_check.py --stt     받아쓰기까지 (모델을 처음 받으면 오래 걸림)
"""

import argparse
import platform
import sys
import time

import voice_io as vio

WAV = "voice_test.wav"


def check_gpu():
    print("\n[2] GPU와 torch")
    try:
        import torch
    except ImportError:
        print("  ✘ torch가 없습니다. 3.12로 실행했는지 확인하세요 (py -3.12 ...)")
        return False
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  ✔ torch {torch.__version__} · {name} · {gb:.1f}GB")
        return True
    print(f"  △ torch {torch.__version__} — GPU를 못 찾습니다. CPU로도 되지만 느립니다.")
    return True


def record_once(rec, seconds_hint="스페이스를 누르고 있는 동안"):
    print(f"\n  {seconds_hint} 녹음합니다. 아무 말이나 한 문장 해 보세요.")
    print("  (예: '천천히 앞으로 숙여서 인사하시오')")
    print("  ...스페이스를 누르세요", end="", flush=True)

    t_wait = time.time()
    while not vio.key_down(vio.VK_SPACE):
        time.sleep(0.02)
        if time.time() - t_wait > 30:
            print("\n  ⚠ 30초 동안 아무 키도 눌리지 않았습니다.")
            return None
    print("\r  ● 녹음 중... (손을 떼면 끝)      ", end="", flush=True)

    rec.start()
    t0 = time.time()
    while vio.key_down(vio.VK_SPACE):
        time.sleep(0.02)
        if time.time() - t0 > 20:
            break
    path = rec.stop(WAV)
    print(f"\r  ■ 녹음 끝 ({time.time() - t0:.1f}초)                 ")
    vio.drain_keys()
    return path


def main():
    ap = argparse.ArgumentParser(description="음성 준비 상태 확인")
    ap.add_argument("--stt", action="store_true", help="받아쓰기까지 시험")
    ap.add_argument("--mic", choices=["auto", "winmm", "sounddevice"], default="auto")
    args = ap.parse_args()

    print(f"\n{platform.system()} · 파이썬 {sys.version.split()[0]}")
    if platform.system() != "Windows":
        print("⚠ 이 프로그램은 윈도우용입니다.")

    print("\n[1] 마이크")
    rec, errors = vio.make_recorder(args.mic)
    for which, e in errors:
        print(f"  · {which} 사용 불가 — {type(e).__name__}: {e}")
    if rec is None:
        print("  ✘ 쓸 수 있는 녹음기가 없습니다.")
        return 1
    print(f"  ✔ {rec.name} 방식으로 녹음합니다"
          f"{' (윈도우 기본 기능, 설치할 것 없음)' if rec.name == 'winmm' else ''}")

    try:
        path = record_once(rec)
    except Exception as e:
        print(f"\n  ✘ 녹음 실패 — {type(e).__name__}: {e}")
        rec.close_quietly()
        return 1
    if path is None:
        rec.close_quietly()
        return 1

    try:
        audio, info = vio.load_wav(path)
    except Exception as e:
        print(f"  ✘ 녹음 파일을 읽지 못했습니다 — {e}")
        return 1
    print(f"  {vio.describe(info)}")
    print(f"  {vio.level_hint(info)}")
    print(f"  파일: {path}   (윈도우 탐색기에서 눌러 직접 들어 보셔도 됩니다)")
    if info["rate"] != 16000 or info["channels"] != 1 or info["bits"] != 16:
        print("  · 형식이 요청과 달라서 여기서 16kHz 모노로 변환했습니다. 문제 없습니다.")

    gpu_ok = check_gpu()

    if not args.stt:
        print("\n" + "─" * 58)
        print("마이크는 됩니다. 받아쓰기까지 보려면:  py -3.12 voice_check.py --stt")
        return 0

    if not gpu_ok:
        return 1

    print("\n[3] 받아쓰기")
    try:
        import stt
        w = stt.Whisper()
        t0 = time.time()
        text = stt.cleanup(w.transcribe(audio))
        print(f"\n  들린 말: 「{text}」" if text else "\n  ⚠ 아무 말도 알아듣지 못했습니다.")
        print(f"  ({time.time() - t0:.1f}초 걸림)")
    except Exception as e:
        print(f"  ✘ {type(e).__name__}: {e}")
        return 1

    print("\n" + "─" * 58)
    if text:
        print("전부 준비됐습니다.  py -3.12 voice_arm.py")
        return 0
    print("녹음은 되는데 말을 못 알아들었습니다.")
    print("  · 위 '녹음 상태' 줄을 보세요. 소리가 작으면 마이크 볼륨을 올리세요.")
    print("  · voice_test.wav를 직접 들어 보시면 원인이 바로 드러납니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
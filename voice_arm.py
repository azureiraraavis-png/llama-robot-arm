# -*- coding: utf-8 -*-
"""
voice_arm.py — 말로 로봇팔을 움직입니다

구조는 지금까지와 똑같습니다. 바뀐 것은 **입구 하나**뿐입니다.

   (전)  키보드로 친 문장 ─┐
   (신)  말 → Whisper ────┴→ 라마 → 검증·보정 → 아두이노

문장이 만들어진 뒤로는 llm_arm_bridge.py의 같은 길을 그대로 지나갑니다.
"n초" 보정도, home 규약도, 각도 제한도 전부 이미 있는 것을 씁니다.
음성이라고 해서 팔의 규칙을 새로 만들지 않습니다 — 규칙이 두 곳에 있으면
반드시 어긋난다는 것을 우리는 이미 겪었으니까요.

준비
  py -3.12 voice_check.py --stt      ← 먼저 이걸로 확인
  ollama serve                        ← 라마가 떠 있어야 합니다

실행
  py -3.12 voice_arm.py                    COM4
  py -3.12 voice_arm.py --dry-run          팔 없이 (말 → 계획까지만 확인)
  py -3.12 voice_arm.py --stt-model openai/whisper-small    가벼운 모델
  py -3.12 voice_arm.py --go               확인 없이 바로 실행

쓰는 법
  스페이스를 누르고 있는 동안 말하고, 손을 뗍니다.
  들린 문장이 화면에 뜹니다.
     Enter  그대로 실행        r  다시 말하기
     e      글자로 고쳐서 실행  n  버리기
  ESC 또는 q + Enter 로 종료합니다.
"""

import argparse
import json
import sys
import time

import llm_arm_bridge as bridge
import stt
import voice_io as vio


def listen_once(rec, max_seconds=20.0):
    """스페이스를 누를 때까지 기다렸다가, 누르고 있는 동안 녹음합니다.
    ESC가 눌리면 None을 돌려줍니다(종료 신호)."""
    print("\n🎤 스페이스를 누르고 말하세요 (ESC=종료)", end="", flush=True)
    while not vio.key_down(vio.VK_SPACE):
        if vio.key_down(vio.VK_ESCAPE):
            print()
            return None
        time.sleep(0.02)

    print("\r● 듣는 중...                                  ", end="", flush=True)
    rec.start()
    t0 = time.time()
    while vio.key_down(vio.VK_SPACE) and time.time() - t0 < max_seconds:
        time.sleep(0.02)
    path = rec.stop("voice_last.wav")
    held = time.time() - t0
    print(f"\r■ {held:.1f}초 들었습니다                        ")
    vio.drain_keys()
    return path


def run_instruction(ser, model, text, history, dry):
    """지금까지 쓰던 그 길. 여기서 새로 판단하는 것은 없습니다."""
    import requests
    try:
        commands = bridge.ask_llama(model, text, history)
        print(f"[계획] {json.dumps(commands, ensure_ascii=False)}")
        commands = bridge.enforce_duration(text, commands)
        commands = bridge.enforce_home_policy(text, commands)
        lines = bridge.validate(commands)
    except requests.ConnectionError:
        print("⚠ Ollama에 연결하지 못했습니다 — 'ollama serve'가 떠 있는지 확인하세요.")
        return
    except Exception as e:
        print(f"⚠ 변환/검증 실패: {e}")
        return

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant",
                    "content": json.dumps(commands, ensure_ascii=False)})

    if dry:
        print("[dry-run] 전송 생략")
        return
    bridge.send_serial(ser, lines)
    n = bridge.dataset_append(text, commands)
    print(f"[기록됨] 데이터셋 {n}건")


def main():
    ap = argparse.ArgumentParser(description="말로 움직이는 로봇팔")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama 모델")
    ap.add_argument("--stt-model", default=stt.DEFAULT_MODEL)
    ap.add_argument("--mic", choices=["auto", "winmm", "sounddevice"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="팔 없이 계획만")
    ap.add_argument("--go", action="store_true", help="확인 없이 바로 실행")
    args = ap.parse_args()

    if not stt.python_ok():
        print("\n" + stt.wrong_python("torch"))
        return 1

    rec, errors = vio.make_recorder(args.mic)
    if rec is None:
        print("⚠ 마이크를 쓸 수 없습니다:")
        for which, e in errors:
            print(f"   {which}: {type(e).__name__}: {e}")
        print("   py -3.12 voice_check.py 로 자세히 확인하세요.")
        return 1
    print(f"[마이크] {rec.name}")

    if not bridge.check_model(args.model):
        rec.close_quietly()
        return 1

    try:
        whisper = stt.Whisper(args.stt_model)
    except RuntimeError as e:
        print(f"\n{e}")
        rec.close_quietly()
        return 1
    except Exception as e:
        print(f"⚠ 음성 인식 모델을 준비하지 못했습니다 — {type(e).__name__}: {e}")
        rec.close_quietly()
        return 1

    ser = None
    if not args.dry_run:
        try:
            import serial
            ser = serial.Serial(args.port, 115200, timeout=0.5)
            time.sleep(2.5)
            ser.reset_input_buffer()
            print(f"[연결됨] {args.port}")
        except Exception as e:
            print(f"⚠ {args.port} 연결 실패: {e}")
            print("  아두이노 IDE의 시리얼 모니터가 열려 있으면 닫으세요. (--dry-run 도 가능)")
            rec.close_quietly()
            return 1

    auto = args.go
    history = []
    print("\n" + "─" * 58)
    print("준비됐습니다." + ("  (바로 실행 모드)" if auto else "  (Enter로 확인 후 실행)"))

    while True:
        try:
            path = listen_once(rec)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠ 녹음 실패 — {type(e).__name__}: {e}")
            break
        if path is None:
            break

        try:
            audio, info = vio.load_wav(path)
        except Exception as e:
            print(f"⚠ 녹음 파일을 읽지 못했습니다 — {e}")
            continue
        if info["seconds"] < 0.3 or info["level"] < 0.005:
            print(f"   {vio.level_hint(info)}")
            continue

        text = stt.cleanup(whisper.transcribe(audio))
        if not text:
            print("   알아듣지 못했습니다. 다시 말해 주세요.")
            continue

        print(f"\n   들린 말: 「{text}」")

        if not auto:
            vio.drain_keys()
            try:
                ans = input("   Enter=실행  r=다시  e=고쳐쓰기  n=버림  q=종료 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            low = ans.lower()
            if low in ("q", "quit", "종료"):
                break
            if low == "n":
                continue
            if low == "r":
                continue
            if low == "e":
                try:
                    edited = input("   고쳐 쓰기> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not edited:
                    continue
                text = edited
            elif low == "go":
                auto = True
                print("   이제부터 확인 없이 바로 실행합니다. (ask 로 되돌리기)")
            elif low == "ask":
                auto = False
                continue
            elif ans:
                print("   모르는 답이라 실행하지 않았습니다.")
                continue

        run_instruction(ser, args.model, text, history, args.dry_run)

    rec.close_quietly()
    if ser:
        bridge.send_serial(ser, ["HOME"])
        ser.close()
    print("\n종료합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
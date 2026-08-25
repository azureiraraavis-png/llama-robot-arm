# -*- coding: utf-8 -*-
"""
arm_tuned.py — 학습된 모델(arm-lora)로 로봇팔을 직접 조종

이번 판의 변경점: 계획을 실행하기 전에 두 가지를 시스템이 보정한다.
  · enforce_duration   지시문의 "n초"와 계획의 대기 시간이 어긋나면 지시문 쪽으로 맞춤
  · enforce_home_policy "유지하라"인데 home으로 끝내거나, "복구하라"인데 home이 없으면 교정
두 함수 모두 llm_arm_bridge.py에 있다 — 팔의 의미를 다루는 코드는 한 곳에만 둔다.

준비:
  py -3.12 -m pip install pyserial
  ※ 반드시 3.12로 실행 (torch가 3.12에만 설치돼 있음)

실행:
  py -3.12 arm_tuned.py                 # COM4, arm-lora
  py -3.12 arm_tuned.py --port COM5
  py -3.12 arm_tuned.py --dry-run       # 팔 없이 계획만 확인
"""

import argparse
import json
import os
import sys
import time

import llm_arm_bridge as bridge          # 검증·보정·전송·기록 재사용
from arm_common import to_messages
from evaluate import parse_output        # 모델 출력 → 명령 리스트

DEFAULT_BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


def load_model(adapter, base_model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"[모델] {adapter} 불러오는 중... (30초쯤 걸립니다)")
    tok = AutoTokenizer.from_pretrained(adapter)
    kw = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={"": 0},
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, **kw)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    try:
        model.generation_config.max_length = None  # max_new_tokens 충돌 경고 제거
    except Exception:
        pass
    print(f"[모델] 준비 완료 ({torch.cuda.get_device_name(0)})")
    return tok, model, torch


def plan(tok, model, torch, instruction):
    """자연어 지시 → 명령 리스트. 온도 0(그리디)이라 같은 지시엔 같은 계획이 나온다."""
    text_in = tok.apply_chat_template(
        to_messages(instruction), tokenize=False, add_generation_prompt=True)
    enc = tok(text_in, add_special_tokens=False, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    took = time.time() - t0
    raw = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return parse_output(raw), raw, took


def main():
    ap = argparse.ArgumentParser(description="학습된 모델로 로봇팔 조종")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--adapter", default="arm-lora")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true", help="팔 없이 계획만 확인")
    ap.add_argument("--dataset", default="arm_dataset_v2.jsonl",
                    help="이번 세션의 기록 파일 (기존 데이터와 섞이지 않게 분리)")
    ap.add_argument("--no-guard", action="store_true",
                    help="시간·home 자동 보정을 끈다 (모델의 원출력을 그대로 보고 싶을 때)")
    args = ap.parse_args()

    bridge.DATASET_FILE = os.path.abspath(args.dataset)

    tok, model, torch = load_model(args.adapter, args.base_model)

    ser = None
    if not args.dry_run:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.5)
        time.sleep(2.5)
        ser.reset_input_buffer()
        print(f"[연결됨] {args.port} @115200")

    print("\n자연어로 지시하세요. (종료: quit / 직전 기록 취소: bad / 직접 명령: ! / 비상 정지: stop)")
    if args.no_guard:
        print("※ 자동 보정 꺼짐 — 모델 원출력 그대로 실행합니다.\n")
    else:
        print("※ 시간·복귀 보정 켜짐 — 보정이 일어나면 ⏱ / ↺ 표시가 뜹니다.\n")

    while True:
        try:
            user_text = input("명령> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "종료"):
            break
        if user_text.lower() == "stop":
            if ser:
                bridge.send_serial(ser, ["HOME"])
            continue
        if user_text.startswith("!"):
            raw = user_text[1:].strip()
            if ser:
                bridge.send_serial(ser, [raw])
            else:
                print(f"[dry-run] {raw}")
            continue
        if user_text.lower() == "bad":
            n = bridge.dataset_remove_last()
            print("삭제할 기록이 없습니다." if n < 0 else f"마지막 기록을 삭제했습니다. ({n}건)")
            continue

        commands, raw, took = plan(tok, model, torch, user_text)
        if commands is None:
            print(f"⚠ 모델 출력을 이해하지 못했습니다 ({took:.1f}초)")
            print(f"   원문: {raw[:200]}")
            continue
        print(f"[계획 {took:.1f}초] {json.dumps(commands, ensure_ascii=False)}")

        if not args.no_guard:
            commands = bridge.enforce_duration(user_text, commands)
            commands = bridge.enforce_home_policy(user_text, commands)

        try:
            lines = bridge.validate(commands)
        except Exception as e:
            print(f"⚠ 검증 실패: {e}")
            continue

        if ser:
            bridge.send_serial(ser, lines)
            n = bridge.dataset_append(user_text, commands)
            print(f"[기록됨] {args.dataset} {n}건 (이상했다면 'bad')")
        else:
            print("[dry-run] 전송 생략")

    if ser:
        bridge.send_serial(ser, ["HOME"])
        ser.close()
    print("종료합니다.")


if __name__ == "__main__":
    sys.exit(main())
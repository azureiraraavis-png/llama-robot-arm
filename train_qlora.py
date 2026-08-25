# -*- coding: utf-8 -*-
"""
train_qlora.py — Llama 3.1 8B를 로봇팔 전용으로 QLoRA 학습 (RTX 4070 12GB)

QLoRA = 모델은 4비트로 압축해 얼려두고(Quantized), 작은 어댑터(LoRA)만 학습.
80억 개 파라미터 중 실제로 학습되는 건 약 0.5%뿐이라 12GB에서도 돌아간다.

사용:
  python train_qlora.py                    # 기본 3에폭
  python train_qlora.py --epochs 5         # 데이터가 적을 땐 에폭을 늘림
  python train_qlora.py --resume           # 중단된 학습 이어서

출력: ./arm-lora/ (학습된 어댑터)
"""

import argparse
import inspect
import json
import math
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from arm_common import to_messages

# 게이트가 없는 4비트 사전양자화 미러 — HF 토큰 없이 바로 받아진다 (약 5.7GB)
DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_LEN = 1024


def encode_ids(tok, messages, add_generation_prompt=False):
    """메시지를 정수 토큰 리스트로 변환.

    transformers 버전마다 apply_chat_template(tokenize=True)의 반환형이 다르다
    (리스트 / BatchEncoding / Encoding 객체). 그래서 템플릿은 '문자열'로만 받고,
    토큰화는 별도로 한다 — 어느 버전에서도 같은 결과가 나온다.
    """
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    # 템플릿이 이미 BOS를 넣으므로 특수토큰 중복 추가를 막는다
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):  # 배치로 감싸여 나온 경우
        ids = ids[0]
    return [int(x) for x in ids]


class ArmDataset(Dataset):
    """지시문은 마스킹하고 정답(JSON)에 대해서만 손실을 계산한다.
    이렇게 해야 모델이 '질문을 흉내내는 법'이 아니라 '답하는 법'을 배운다."""

    def __init__(self, path, tokenizer):
        self.tok = tokenizer
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        full = to_messages(r["instruction"], r["output"])
        prompt_only = full[:2]

        ids_prompt = encode_ids(self.tok, prompt_only, add_generation_prompt=True)
        ids_full = encode_ids(self.tok, full)

        # 안전장치: 프롬프트가 전체의 접두사가 아니면 마스킹이 틀어진다
        if ids_full[: len(ids_prompt)] != ids_prompt:
            n_prompt = min(len(ids_prompt), len(ids_full) - 1)
        else:
            n_prompt = len(ids_prompt)

        ids_full = ids_full[:MAX_LEN]
        labels = list(ids_full)
        for k in range(min(n_prompt, len(labels))):
            labels[k] = -100  # 지시문 부분은 학습에서 제외

        return {"input_ids": ids_full, "labels": labels}


def build_training_args(**wanted):
    """설치된 transformers 버전이 받는 인자만 골라서 TrainingArguments를 만든다.
    (4.x와 5.x에서 인자 이름이 달라 그대로 넘기면 TypeError가 난다)"""
    params = inspect.signature(TrainingArguments.__init__).parameters
    accepted = set(params)
    # **kwargs를 받는 시그니처면 이름 검사를 신뢰할 수 없으므로 필터링하지 않는다
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if takes_kwargs:
        wanted.pop("_total_steps", None)
        return TrainingArguments(**wanted)

    # 5.x에서 사라진 warmup_ratio → warmup_steps로 환산
    if "warmup_ratio" in wanted and "warmup_ratio" not in accepted:
        ratio = wanted.pop("warmup_ratio")
        total = wanted.pop("_total_steps", 0)
        if "warmup_steps" in accepted and total:
            wanted["warmup_steps"] = max(1, int(total * ratio))
    wanted.pop("_total_steps", None)

    dropped = [k for k in wanted if k not in accepted]
    for k in dropped:
        wanted.pop(k)
    if dropped:
        print(f"[호환] 이 transformers 버전이 받지 않는 인자는 생략: {', '.join(dropped)}")
    return TrainingArguments(**wanted)


def load_base_model(name, quant_cfg):
    """transformers 5.x는 torch_dtype 대신 dtype을 쓴다."""
    kw = dict(quantization_config=quant_cfg, device_map={"": 0})
    key = "dtype" if "dtype" in inspect.signature(
        AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    kw[key] = torch.bfloat16
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:
        kw.pop(key, None)
        return AutoModelForCausalLM.from_pretrained(name, **kw)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--out", default="arm-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("⚠ CUDA를 찾지 못했습니다. GPU 드라이버와 PyTorch 설치를 확인하세요.")
        print("   (CPU로는 8B 학습이 사실상 불가능합니다)")
        return
    print(f"[GPU] {torch.cuda.get_device_name(0)} / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[모델] {args.model} 로딩 중... (처음 실행 시 다운로드에 시간이 걸립니다)")
    model = load_base_model(args.model, BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    ds = ArmDataset(args.train, tok)

    # 사전 점검: 토큰화 결과가 정수 리스트인지 여기서 확인한다
    # (학습 도중에 터지면 몇 분을 낭비하므로 시작 전에 걸러낸다)
    probe = ds[0]
    bad = [k for k, v in probe.items()
           if not isinstance(v, list) or not all(isinstance(x, int) for x in v)]
    if bad:
        print(f"⚠ 토큰화 결과가 정수 리스트가 아닙니다: {bad}")
        print("  transformers 버전 문제일 수 있습니다. 이 메시지를 그대로 알려주세요.")
        return
    n_train = sum(1 for x in probe["labels"] if x != -100)
    print(f"[점검] 첫 샘플 토큰 {len(probe['input_ids'])}개 중 학습 대상 {n_train}개 "
          f"(나머지는 지시문이라 마스킹됨) — 정상")

    accum = 8
    steps_per_epoch = max(1, math.ceil(len(ds) / accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"[데이터] 학습 {len(ds)}건 · 총 {total_steps}스텝 (에폭당 {steps_per_epoch})")

    trainer = Trainer(
        model=model,
        args=build_training_args(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=accum,   # 실효 배치 8
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            _total_steps=total_steps,            # warmup_steps 환산에 사용
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to=[],
            seed=42,
        ),
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n✅ 학습 완료 → {os.path.abspath(args.out)}")
    print("   다음: python evaluate.py --tuned  로 성적을 확인하세요")


if __name__ == "__main__":
    main()
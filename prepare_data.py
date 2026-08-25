# -*- coding: utf-8 -*-
"""
prepare_data.py — 정제본을 학습/평가 세트로 분할

평가 세트는 학습에 절대 쓰지 않는다. 그래야 "외운 것"이 아니라
"배운 것"을 측정할 수 있다. 유형별로 골고루 섞이도록 나눈다.

사용: python prepare_data.py arm_dataset_clean.jsonl
출력: train.jsonl, eval.jsonl
"""

import json
import random
import re
import sys
from collections import defaultdict

SEED = 42
EVAL_RATIO = 0.15


def kind(inst: str) -> str:
    """지시문을 유형별로 분류 (평가 세트에 유형이 골고루 들어가게)"""
    if re.search(r"\d+\s*도", inst) and re.search(r"(기울|굽히|접)", inst):
        return "상대각도"
    if re.search(r"\d+\s*초", inst):
        return "시간"
    if re.search(r"\d+\s*번", inst):
        return "횟수"
    if re.search(r"(그대로|유지|멈추|복구하지)", inst):
        return "자세유지"
    if re.search(r"(집|잡|시늉)", inst):
        return "파지"
    if re.search(r"(인사|도리도리|두리번|춤)", inst):
        return "제스처"
    return "기타"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "arm_dataset_clean.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

    buckets = defaultdict(list)
    for r in rows:
        buckets[kind(r["instruction"])].append(r)

    rng = random.Random(SEED)
    train, ev = [], []
    for k, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * EVAL_RATIO))
        ev += items[:n_eval]
        train += items[n_eval:]
        print(f"  {k:<8} 전체 {len(items):>3}건 → 학습 {len(items)-n_eval:>3} / 평가 {n_eval}")

    rng.shuffle(train)
    for name, data in (("train.jsonl", train), ("eval.jsonl", ev)):
        with open(name, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n학습 {len(train)}건 / 평가 {len(ev)}건 저장 완료")


if __name__ == "__main__":
    main()
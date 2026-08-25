# -*- coding: utf-8 -*-
"""
arm_common.py — 학습·평가가 공유하는 정의

핵심 아이디어: 파인튜닝의 목적은 지금까지 시스템 프롬프트에 쌓아온
규칙 9개와 예시 15개를 '모델 가중치 안으로' 옮기는 것이다.
따라서 학습에는 규칙과 예시를 뺀 짧은 프롬프트만 쓴다.
(프롬프트가 짧아지면 응답도 빨라지고, 컨텍스트 여유도 생긴다)
"""

import json
import re

NUM_JOINTS = 5
JOINT_LIMITS = [(0, 180), (15, 165), (15, 165), (10, 170), (30, 125)]

# 학습·추론에 공통으로 쓰는 짧은 프롬프트 (규칙/예시 없음 — 그건 학습으로 배운다)
SHORT_SYSTEM = """당신은 5축 로봇팔 제어기입니다. 지시를 {"commands":[...]} JSON으로만 출력합니다.

관절: 0=베이스회전(0~180, 90=정면, 클수록 왼쪽) 1=어깨(15~165, 작을수록 앞으로)
      2=팔꿈치(15~165) 3=손목회전(10~170) 4=집게(grip으로 제어)

명령: {"cmd":"home"} | {"cmd":"joint","joint":j,"angle":a} | {"cmd":"move","joint":j,"delta":d}
      {"cmd":"grip","close":true} | {"cmd":"wait","seconds":n} |{"cmd":"wait","ms":n} | {"cmd":"speed","level":1~10}
      {"cmd":"repeat","times":n,"commands":[...]} | {"cmd":"repeat","seconds":n,"commands":[...]}
      
      {"cmd":"dance","routine":"wave"} | routine: all bow wave sway shake clap robot sweep finale
      선택: "bpm":140 "amp":0.6 "grip":false"""


def to_messages(instruction: str, output: dict = None) -> list:
    msgs = [
        {"role": "system", "content": SHORT_SYSTEM},
        {"role": "user", "content": instruction},
    ]
    if output is not None:
        msgs.append({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)})
    return msgs


# ── 채점용 유틸 ───────────────────────────────────────────────────────────
def expand(commands: list) -> list:
    """repeat를 펼쳐 실제 실행될 명령 목록으로 만든다 (횟수 비교용)."""
    out = []
    for c in commands:
        if not isinstance(c, dict):
            continue
        if c.get("cmd") == "repeat":
            n = int(c.get("times", 1)) if "times" in c else 1
            for _ in range(max(1, min(20, n))):
                out.extend(expand(c.get("commands", [])))
        else:
            out.append(c)
    return out


def is_valid(commands) -> bool:
    """브리지가 실행할 수 있는 형태인지 검사 (검증기의 축약판)."""
    if not isinstance(commands, list) or not commands:
        return False
    ok = {"home", "joint", "move", "grip", "wait", "speed", "repeat"}
    for c in commands:
        if not isinstance(c, dict) or c.get("cmd") not in ok:
            return False
        cmd = c["cmd"]
        if cmd == "joint":
            if not isinstance(c.get("angle"), int) or not (0 <= c.get("joint", -1) < NUM_JOINTS):
                return False
        elif cmd == "move":
            if not isinstance(c.get("delta"), int) or c.get("delta") == 0:
                return False
            if not (0 <= c.get("joint", -1) < NUM_JOINTS):
                return False
        elif cmd == "grip":
            if not isinstance(c.get("close"), bool):
                return False
        elif cmd == "repeat":
            if not c.get("commands") or not is_valid(c["commands"]):
                return False
            if "times" not in c and "seconds" not in c:
                return False
    return True


GESTURE = re.compile(r"(인사|시늉|집|잡|도리도리|두리번|춤|딱딱|흔들|부딪)")
COUNT = re.compile(r"(\d+)\s*번")
DEG = re.compile(r"(-?\d+)\s*도")


def applicable(instruction: str, ref: list) -> list:
    """이 문제에 어떤 채점 기준이 적용되는지 — 예측과 무관하게 문제와 정답만 보고 정한다.
    (예측이 파싱조차 안 됐을 때도 같은 분모로 채점하기 위해)"""
    keys = ["valid", "exact"]
    m = COUNT.search(instruction)
    if m and int(m.group(1)) >= 2:
        keys.append("count")
    if ("앞으로" in instruction) ^ ("뒤로" in instruction):
        er = expand(ref)
        moves_shoulder = any(
            (c.get("cmd") == "joint" and c.get("joint") == 1 and c.get("angle") != 90)
            or (c.get("cmd") == "move" and c.get("joint") == 1)
            for c in er
        )
        if moves_shoulder:
            keys.append("direction")
    if DEG.search(instruction) and re.search(r"(기울|굽히|접|숙이|올리|내리)", instruction):
        keys.append("relative")
    if GESTURE.search(instruction) and len(expand(ref)) >= 3:
        keys.append("sequence")
    return keys


def score_one(instruction: str, pred, ref: list) -> dict:
    """예측 하나를 채점. pred가 None(파싱 실패)이면 적용 기준 전부 오답 처리."""
    keys = applicable(instruction, ref)
    if pred is None:
        return {k: False for k in keys}

    s = {}
    ep, er = expand(pred), expand(ref)
    for k in keys:
        if k == "valid":
            s[k] = is_valid(pred)
        elif k == "exact":
            s[k] = json.dumps(pred, sort_keys=True) == json.dumps(ref, sort_keys=True)
        elif k == "count":
            s[k] = len(ep) == len(er)
        elif k == "direction":
            fwd = "앞으로" in instruction
            angles = [c["angle"] for c in ep
                      if c.get("cmd") == "joint" and c.get("joint") == 1 and c.get("angle") != 90]
            deltas = [c["delta"] for c in ep if c.get("cmd") == "move" and c.get("joint") == 1]
            s[k] = bool(angles or deltas) and all((a < 90) == fwd for a in angles) \
                and all((d < 0) == fwd for d in deltas)
        elif k == "relative":
            s[k] = any(c.get("cmd") == "move" for c in ep)
        elif k == "sequence":
            s[k] = len(ep) >= 3
    return s
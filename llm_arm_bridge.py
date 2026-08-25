# -*- coding: utf-8 -*-
"""
llm_arm_bridge.py — Llama(Ollama) ↔ 아두이노 로봇팔 브리지

자연어 명령을 입력하면:
  1) 로컬 Ollama의 Llama 모델이 JSON 명령 시퀀스로 변환
  2) 파이썬이 스키마/한계값 검증 (LLM 출력을 그대로 믿지 않음)
  3) 검증된 명령을 시리얼 텍스트 프로토콜로 아두이노에 전송

사전 준비:
  pip install pyserial requests
  ollama pull llama3.2:3b      (GPU가 있으면 llama3.1:8b 권장)
  아두이노에 robot_arm_llm.ino 업로드

실행:
  python llm_arm_bridge.py --port COM3
  (포트는 장치관리자 또는 아두이노 IDE에서 확인. --dry-run 이면 로봇 없이 LLM만 테스트)
"""

import argparse
import json
import os
import re
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"

# 파인튜닝용 데이터셋 (스크립트와 같은 폴더에 누적)
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_dataset.jsonl")

NUM_JOINTS = 5
# 아두이노 스케치의 JOINT_MIN/JOINT_MAX와 반드시 일치시킬 것
JOINT_LIMITS = [(0, 180), (15, 165), (15, 165), (10, 170), (30, 125)]

SYSTEM_PROMPT = """당신은 5축 로봇팔 제어기입니다. 사용자의 자연어 지시를 {"commands":[명령1, 명령2, ...]} 형태의 JSON 객체 하나로 변환합니다.

로봇 구성:
- 관절 0: 베이스 회전 (0~180, 90=정면, 0=오른쪽 끝, 180=왼쪽 끝)
- 관절 1: 어깨 (15~165, 작을수록 앞으로 숙임)
- 관절 2: 팔꿈치 (15~165)
- 관절 3: 손목 회전 (10~170, 그리퍼를 좌우로 비트는 롤 축. 90=정면. 고개를 끄덕이는 동작은 못 하고, 도리도리/두리번거림/비틀기에 적합)
- 관절 4: 그리퍼 (직접 제어하지 말고 grip 명령 사용)

사용 가능한 명령 (이것만 사용):
  {"cmd":"home"}                                  홈 자세
  {"cmd":"joint","joint":0,"angle":90}            한 관절을 그 각도로 이동 (절대 각도)
  {"cmd":"move","joint":1,"delta":30}             현재 자세에서 상대적으로 이동 (양수=뒤로/위로/왼쪽, 음수=앞으로/아래로/오른쪽)
  {"cmd":"pose","angles":[90,60,120,90,60]}       전체 자세 (5개 각도)
  {"cmd":"grip","close":true}                     그리퍼 닫기(true)/열기(false)
  {"cmd":"wait","ms":500}                         짧은 대기 (밀리초, 동작 사이 간격용)
  {"cmd":"wait","seconds":10}                     n초 대기 — "n초 기다려"는 반드시 이 형태로.
                                                  밀리초로 직접 환산하지 마라(시스템이 한다)
  {"cmd":"speed","level":5}                       속도 1(느림)~10(빠름)
  {"cmd":"repeat","seconds":5,"commands":[...]}   안의 명령들을 n초 동안 반복 (반복 횟수 계산은 시스템이 함)
  {"cmd":"repeat","times":5,"commands":[...]}     안의 명령들을 n번 반복 (횟수 세기는 시스템이 함)
  {"cmd":"dance","routine":"wave"}                미리 짜둔 안무를 춘다. routine은 아래 중 하나:
      all(전곡) bow(인사) wave(웨이브) sway(좌우스윙) shake(도리도리)
      clap(집게박수) robot(로봇춤) sweep(큰스윕) finale(피날레)
      선택: "bpm":140(빠르게) "amp":0.6(작게) "grip":false(집게 빼기)

규칙:
1. 반드시 {"commands":[...]} 객체 하나만 출력. 설명, 인사, 마크다운 금지.
2. 몸짓/제스처/동작 지시는 commands 배열에 항상 3개 이상의 명령을 담는다. 명령 하나만 담는 것은 거의 항상 잘못이다.
3. speed 명령은 단독으로 쓰지 않는다. 반드시 뒤에 움직임 명령이 따라야 하고, 시퀀스 끝에 {"cmd":"speed","level":5}로 복원한다.
4. "잡다/집다/쥐다/줍다"가 들어간 지시(시늉·흉내 포함)는 반드시 이 5단계 전체를 순서대로 포함한다:
   그리퍼 열기 → 앞으로 숙여 접근(어깨/팔꿈치) → 그리퍼 닫기 → 잠시 대기 → 들어올리기.
   grip 명령 하나만 출력하는 것은 오답이다.
5. 각도는 반드시 각 관절의 허용 범위 안에서.
6. 동작이 끝나면 특별한 이유가 없는 한 {"cmd":"home"}으로 마무리한다.
   단, "~한 채로", "~인 상태로 있어", "그대로 유지해"처럼 자세를 유지하라는 지시는 home으로 끝내지 않는다.
   "n도 기울여/올려/내려/돌려" 같은 상대적 이동 지시도 home을 붙이지 않는다 — 붙이면 기울인 자세가 즉시 사라진다.
6-1. "n도 기울여", "n도씩 굽혀", "조금 더 올려", "10도 돌려"처럼 현재 자세를 기준으로 한 증감 지시는
   joint(절대 각도)가 아니라 move를 쓴다. 현재 각도를 추측하거나 더하기를 직접 하지 마라 —
   시스템이 실제 팔에게 현재 각도를 물어보고 계산한다.
6-2. joint의 angle에는 음수를 절대 쓰지 않는다. angle은 0~180 범위의 위치 값이며 음수 위치란 존재하지 않는다.
   음수는 오직 move의 delta에만 쓴다 (delta -15 = 현재보다 15도 앞으로/아래로).
7. 불가능하거나 위험한 지시면 {"commands":[{"cmd":"home"}]} 을 출력.
8. "n초 동안 X 해"라는 지시는 wait가 아니라 repeat를 쓴다: X의 핵심 왕복 동작을 commands에 담고 seconds에 n을 넣는다.
   반복 횟수를 직접 계산하지 마라. 시스템이 시계를 보며 알아서 반복한다.
   "n번 X 해"라는 지시는 repeat의 times에 n을 넣는다. 같은 명령을 n번 직접 나열하며 세지 마라 — 세기는 시스템이 한다.
8-1. 춤 지시("춤춰", "웨이브 해봐", "신나게 움직여")는 dance 명령 하나로 낸다.
   자세를 직접 나열하지 마라 — 안무와 박자는 시스템이 갖고 있다.
   "빠르게"는 bpm을 올리고, "살살/작게"는 amp를 낮추고, 집게가 불안하면 grip:false.
9. 지시가 "~하고", "~한 다음", "~하고 나서", "먼저 ~ 그리고 ~" 처럼 여러 동작을 잇고 있으면:
   동작을 순서대로 나눈 뒤, 각 동작의 전체 시퀀스를 commands 배열에 차례로 이어붙인다.
   중간의 home은 생략해도 되지만 마지막은 home으로 끝낸다. 동작 사이에 {"cmd":"wait","ms":400}를 넣으면 자연스럽다.

예시)
사용자: 오른쪽으로 돌아서 인사해
출력: {"commands":[{"cmd":"joint","joint":0,"angle":30},{"cmd":"joint","joint":1,"angle":55},{"cmd":"wait","ms":600},{"cmd":"joint","joint":1,"angle":90},{"cmd":"home"}]}

사용자: 고개를 도리도리 저어봐
출력: {"commands":[{"cmd":"joint","joint":3,"angle":50},{"cmd":"joint","joint":3,"angle":130},{"cmd":"joint","joint":3,"angle":50},{"cmd":"joint","joint":3,"angle":90},{"cmd":"home"}]}

사용자: 천천히 앞으로 숙여서 인사해
출력: {"commands":[{"cmd":"speed","level":2},{"cmd":"joint","joint":1,"angle":50},{"cmd":"wait","ms":800},{"cmd":"joint","joint":1,"angle":90},{"cmd":"speed","level":5},{"cmd":"home"}]}

사용자: 뭔가 잡는 시늉을 해봐
출력: {"commands":[{"cmd":"grip","close":false},{"cmd":"joint","joint":1,"angle":60},{"cmd":"joint","joint":2,"angle":120},{"cmd":"grip","close":true},{"cmd":"wait","ms":600},{"cmd":"joint","joint":1,"angle":90},{"cmd":"grip","close":false},{"cmd":"home"}]}

사용자: 작은 물체를 잡는 시늉을 하시오
출력: {"commands":[{"cmd":"grip","close":false},{"cmd":"joint","joint":1,"angle":55},{"cmd":"joint","joint":2,"angle":125},{"cmd":"grip","close":true},{"cmd":"wait","ms":800},{"cmd":"joint","joint":1,"angle":90},{"cmd":"joint","joint":2,"angle":90},{"cmd":"grip","close":false},{"cmd":"home"}]}

사용자: 바닥에 있는 걸 주워봐
출력: {"commands":[{"cmd":"grip","close":false},{"cmd":"speed","level":3},{"cmd":"joint","joint":1,"angle":45},{"cmd":"joint","joint":2,"angle":135},{"cmd":"grip","close":true},{"cmd":"wait","ms":500},{"cmd":"joint","joint":1,"angle":90},{"cmd":"speed","level":5},{"cmd":"home"}]}

사용자: 춤 춰봐
출력: {"commands":[{"cmd":"dance","routine":"all"}]}

사용자: 웨이브 한번 해봐
출력: {"commands":[{"cmd":"dance","routine":"wave"}]}

사용자: 신나게 빠른 춤 춰줘
출력: {"commands":[{"cmd":"dance","routine":"all","bpm":150}]}

사용자: 살살 춤춰, 집게는 빼고
출력: {"commands":[{"cmd":"dance","routine":"all","amp":0.6,"grip":false}]}

사용자: 인사하고 나서 로봇춤 춰
출력: {"commands":[{"cmd":"dance","routine":"bow"},{"cmd":"dance","routine":"robot"}]}

사용자: 좌우로 두리번거려
출력: {"commands":[{"cmd":"joint","joint":0,"angle":40},{"cmd":"wait","ms":400},{"cmd":"joint","joint":0,"angle":140},{"cmd":"wait","ms":400},{"cmd":"home"}]}

사용자: 5초 동안 고개를 도리도리 저어봐
출력: {"commands":[{"cmd":"repeat","seconds":5,"commands":[{"cmd":"joint","joint":3,"angle":50},{"cmd":"joint","joint":3,"angle":130}]},{"cmd":"joint","joint":3,"angle":90},{"cmd":"home"}]}

사용자: 팔을 뒤로 30도 기울이시오
출력: {"commands":[{"cmd":"move","joint":1,"delta":30}]}

사용자: 팔을 앞으로 15도 더 숙여
출력: {"commands":[{"cmd":"move","joint":1,"delta":-15}]}

사용자: 모든 관절을 15도씩 굽히시오
출력: {"commands":[{"cmd":"move","joint":1,"delta":-15},{"cmd":"move","joint":2,"delta":-15},{"cmd":"move","joint":3,"delta":-15}]}

사용자: 왼쪽으로 20도 돌린 채 3초 기다려
출력: {"commands":[{"cmd":"move","joint":0,"delta":20},{"cmd":"wait","ms":3000}]}

사용자: 집게를 5번 딱딱거려
출력: {"commands":[{"cmd":"repeat","times":5,"commands":[{"cmd":"grip","close":true},{"cmd":"grip","close":false}]},{"cmd":"home"}]}

사용자: 3번 꾸벅 인사해
출력: {"commands":[{"cmd":"repeat","times":3,"commands":[{"cmd":"joint","joint":1,"angle":55},{"cmd":"joint","joint":1,"angle":90}]},{"cmd":"home"}]}

사용자: 10초 동안 좌우로 두리번거려
출력: {"commands":[{"cmd":"repeat","seconds":10,"commands":[{"cmd":"joint","joint":0,"angle":40},{"cmd":"joint","joint":0,"angle":140}]},{"cmd":"home"}]}

사용자: 왼쪽을 본 다음 앞으로 숙여 인사해
출력: {"commands":[{"cmd":"joint","joint":0,"angle":150},{"cmd":"wait","ms":400},{"cmd":"joint","joint":1,"angle":55},{"cmd":"wait","ms":600},{"cmd":"joint","joint":1,"angle":90},{"cmd":"home"}]}

사용자: 고개를 도리도리 젓고 나서 뭔가 잡는 시늉을 해
출력: {"commands":[{"cmd":"joint","joint":3,"angle":50},{"cmd":"joint","joint":3,"angle":130},{"cmd":"joint","joint":3,"angle":90},{"cmd":"wait","ms":400},{"cmd":"grip","close":false},{"cmd":"joint","joint":1,"angle":60},{"cmd":"joint","joint":2,"angle":120},{"cmd":"grip","close":true},{"cmd":"wait","ms":600},{"cmd":"joint","joint":1,"angle":90},{"cmd":"grip","close":false},{"cmd":"home"}]}
"""


def list_models():
    """Ollama에 설치된 모델 이름 목록. 연결 실패면 None."""
    try:
        r = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=10)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return None


def check_model(model: str) -> bool:
    """모델이 실제로 있는지 미리 확인한다.
    (없으면 매 명령마다 404가 뜨는데, 그 메시지만으로는 원인을 알기 어렵다)"""
    names = list_models()
    if names is None:
        print("⚠ Ollama에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
        return False
    if model in names or any(n.split(":")[0] == model.split(":")[0] for n in names):
        return True
    print(f"⚠ 모델 '{model}' 을(를) 찾을 수 없습니다.")
    if names:
        print("   설치된 모델:")
        for n in names:
            print(f"     · {n}")
        print(f"   → 위 중 하나를 쓰려면:  --model <이름>")
    else:
        print("   설치된 모델이 없습니다.")
    print(f"   → 새로 받으려면:  ollama pull {model}")
    return False


def ask_llama(model: str, user_text: str, history: list) -> list:
    """Ollama에 질의해 명령 리스트(JSON)를 얻는다."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # 최근 대화 일부만 유지
    messages.append({"role": "user", "content": user_text})

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",  # JSON 강제 (Ollama 기능)
            # repeat_penalty=1.0: 같은 지시를 연달아 내려도 같은 계획을 다시 출력할 수 있게
            # 반복 억제를 끈다 (구조화된 명령 출력에는 반복 페널티가 해롭다)
            # num_ctx=8192: Ollama 기본 컨텍스트(2048)로는 긴 시스템 프롬프트 + 대화 기억이
            # 넘쳐서 규칙 앞부분이 잘려나갈 수 있다 (12GB VRAM에서 8192는 여유)
            "options": {"temperature": 0.2, "num_predict": 512, "repeat_penalty": 1.0, "num_ctx": 8192},
        },
        timeout=300,
    )
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama가 모델 '{model}'을(를) 찾지 못했습니다 (404). "
            f"'ollama list'로 확인하고 'ollama pull {model}' 하거나 --model 로 바꾸세요.\n"
            f"   서버 응답: {resp.text[:160]}")
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()

    # 모델이 배열 대신 객체로 감싸거나 코드블록을 붙이는 경우 방어
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
    data = json.loads(content)
    if isinstance(data, dict):
        # {"commands":[...]} 또는 단일 명령 형태 방어
        data = data.get("commands", [data])
    if not isinstance(data, list):
        raise ValueError("LLM 출력이 JSON 배열이 아닙니다")
    return data


def validate(commands: list, allow_repeat: bool = True) -> list:
    """LLM 출력 검증: 허용된 명령/범위만 통과시키고 시리얼 라인으로 변환.
    repeat는 {"repeat_s": 초, "lines": [...]} 딕셔너리로 변환되어 전송 단계에서 시간 반복된다."""
    lines = []
    for i, c in enumerate(commands):
        if not isinstance(c, dict) or "cmd" not in c:
            raise ValueError(f"{i}번째 명령 형식 오류: {c}")
        cmd = c["cmd"]

        if cmd == "repeat":
            if not allow_repeat:
                raise ValueError("repeat 안에 repeat를 중첩할 수 없습니다")
            sub = validate(c.get("commands", []), allow_repeat=False)
            if not sub:
                raise ValueError("repeat의 commands가 비어 있습니다")
            if "times" in c:  # 횟수 반복: 파이썬이 정확히 n번 펼친다
                n = max(1, min(20, int(c["times"])))
                for _ in range(n):
                    lines.extend(sub)
            else:  # 시간 반복: 전송 단계에서 실제 시계로 반복
                secs = max(1, min(30, int(c.get("seconds", 5))))
                lines.append({"repeat_s": secs, "lines": sub})
        elif cmd == "dance":
            # 모델은 "어떤 춤"만 말하고, 안무·박자·속도 계산은 dance.py가 한다
            # (repeat의 횟수 세기, wait의 초 환산과 같은 구조)
            import dance
            keys = dance.resolve(c.get("routine"))
            if keys is None:
                raise ValueError(f"모르는 안무: {c.get('routine')} "
                                 f"(가능: {', '.join(dance.ORDER)}, all)")
            lines.extend(dance.to_lines(
                keys,
                bpm=c.get("bpm", 100),
                amp=c.get("amp", 1.0),
                use_grip=bool(c.get("grip", True)),
            ))
        elif cmd == "home":
            lines.append("HOME")
        elif cmd == "move":
            # 상대 이동: 전송 단계에서 실제 팔의 현재 각도를 물어본 뒤 계산한다
            j = int(c["joint"])
            if not (0 <= j < NUM_JOINTS):
                raise ValueError(f"관절 번호 범위 초과: {j}")
            d = max(-180, min(180, int(c.get("delta", 0))))
            if d == 0:
                raise ValueError("move의 delta가 0입니다")
            lines.append({"rel_joint": j, "delta": d})
        elif cmd == "joint":
            j = int(c["joint"])
            a = int(c["angle"])
            if not (0 <= j < NUM_JOINTS):
                raise ValueError(f"관절 번호 범위 초과: {j}")
            if a < 0:
                # 음수 절대각은 존재하지 않는다 → 상대 이동을 의도한 것으로 해석
                print(f"  ↺ 관절 {j}의 각도 {a}는 음수 → 상대 이동({a}도)으로 해석합니다")
                lines.append({"rel_joint": j, "delta": max(-180, a)})
                continue
            lo, hi = JOINT_LIMITS[j]
            if not (lo <= a <= hi):
                print(f"  ⚠ 관절 {j}의 각도 {a} → 안전 범위({lo}~{hi})로 조정")
                a = max(lo, min(hi, a))
            lines.append(f"S {j} {a}")
        elif cmd == "pose":
            angles = c["angles"]
            if not (isinstance(angles, list) and len(angles) == NUM_JOINTS):
                raise ValueError(f"pose는 각도 {NUM_JOINTS}개 필요: {angles}")
            clamped = [
                max(lo, min(hi, int(a)))
                for a, (lo, hi) in zip(angles, JOINT_LIMITS)
            ]
            lines.append("P " + " ".join(str(a) for a in clamped))
        elif cmd == "grip":
            lines.append(f"GRIP {1 if c.get('close') else 0}")
        elif cmd in ("wait", "hold"):
            # seconds로 오면 밀리초 환산은 여기서 한다 — 모델에게 곱셈을 시키지 않는다
            if "seconds" in c:
                ms = int(float(c["seconds"]) * 1000)
            else:
                ms = int(c.get("ms", 500))
            # 총 대기는 최대 60초까지 허용하되, 아두이노 규격(1회 5초)에 맞게 조각내서 전송
            ms = max(0, min(60000, ms))
            while ms > 5000:
                lines.append("WAIT 5000")
                ms -= 5000
            if ms > 0:
                lines.append(f"WAIT {ms}")
        elif cmd == "speed":
            lv = max(1, min(10, int(c.get("level", 5))))
            lines.append(f"SPEED {lv}")
        else:
            raise ValueError(f"허용되지 않은 명령: {cmd}")
    return lines


DURATION_RE = re.compile(r"\d+\s*초")
# "기다려/대기/가만히 있어" — 시간이 붙으면 그 시간만, 시간이 없으면 무기한
WAIT_VERB_RE = re.compile(r"(기다리|기다려|대기|가만히\s*있)")
# "유지/멈춰/계속" — 시간이 붙어 있어도 자세를 유지하라는 뜻
HOLD_STRONG_RE = re.compile(r"(유지|멈추|멈춰|계속)")
# "그대로/그 상태로/그 자세로" — 단독이면 유지, 시간+대기동사와 함께면 그 시간만
HOLD_SOFT_RE = re.compile(r"(그대로|그\s*상태|그\s*자세|그\s*자리|이\s*상태|이\s*자세)")
NO_RETURN_RE = re.compile(r"(원상복구|원상복귀|복구)\s*하지\s*(마|말)")
RETURN_RE = re.compile(r"(원상복구|원상복귀|원래\s*상태로\s*돌아|원위치로\s*되돌)")


def home_policy(instruction: str):
    """이 지시가 home으로 끝나야 하는가. True/False, 판단 불가면 None.

    ★ 이 함수가 home 규약의 유일한 정의다. 검사기(check_dataset.py)와
      정제기(fix_dataset_v2.py)도 여기서 가져다 쓴다 —
      규칙이 두 곳에 복사되면 반드시 어긋난다(그래서 오늘 고생했다).

    규약 (우선순위 순)
      1. "원상복구 하지 마/말"              → home 없음
      2. "원상복구/원상복귀/원래 상태로"     → home 있음
      3. "유지/멈춰/계속"                   → home 없음 (시간이 있어도 유지)
      4. "n초" + "기다려/대기/가만히 있어"   → home 있음 (그 시간만 대기 후 복귀)
      5. 시간 없는 "기다려" 또는 "그대로/그 상태로" → home 없음 (무기한 유지)
    """
    i = instruction
    if NO_RETURN_RE.search(i):
        return False
    if RETURN_RE.search(i):
        return True
    if HOLD_STRONG_RE.search(i):
        return False
    if DURATION_RE.search(i) and WAIT_VERB_RE.search(i):
        return True
    if WAIT_VERB_RE.search(i) or HOLD_SOFT_RE.search(i):
        return False
    return None


def enforce_home_policy(instruction: str, commands: list) -> list:
    """지시문과 계획의 끝 home이 어긋나면 바로잡는다.

    모델이 "그대로 멈추시오"라고 했는데 home으로 끝내면 지시를 정면으로 뒤집는 것이고,
    "원상복구 하시오"인데 home이 없으면 요청을 무시하는 것이다.
    각도 한계를 아두이노가 강제하듯, 이 의도도 코드가 지킨다.

    규약은 home_policy()에 정의돼 있다. 판단 근거가 없으면 모델의 판단을 존중한다.
    """
    if not commands:
        return commands
    want = home_policy(instruction)
    if want is None:
        return commands
    has = commands[-1].get("cmd") == "home"

    if want and not has:
        print("  ↺ 복귀 지시가 있는데 계획에 home이 없어 추가합니다")
        return commands + [{"cmd": "home"}]
    if has and not want:
        print("  ↺ 자세를 유지하라는 지시라 끝의 home을 제거합니다")
        return commands[:-1]
    return commands


def _duration_slots(commands, path=()):
    """계획 안에서 시간을 나타내는 자리를 순서대로 모두 찾는다.
    (wait/hold의 대기, repeat의 seconds)"""
    slots = []
    for k, c in enumerate(commands):
        if not isinstance(c, dict):
            continue
        cmd = c.get("cmd")
        if cmd in ("wait", "hold"):
            ms = int(float(c["seconds"]) * 1000) if "seconds" in c else int(c.get("ms", 0))
            slots.append((path + (k,), ms))
        elif cmd == "repeat":
            if "seconds" in c:
                slots.append((path + (k,), int(c["seconds"]) * 1000))
            slots += _duration_slots(c.get("commands", []), path + (k, "commands"))
    return slots


def _set_duration(commands, path, ms):
    node = commands
    for p in path[:-1]:
        node = node[p] if isinstance(p, int) else node[p]
    c = node[path[-1]]
    secs = max(1, round(ms / 1000))
    c.pop("seconds", None)
    if c.get("cmd") != "repeat":
        c.pop("ms", None)
    c["seconds"] = secs


def enforce_duration(instruction: str, commands: list) -> list:
    """지시문의 "n초"와 계획의 대기 시간이 어긋나면 지시문 쪽으로 맞춘다.

    언어 모델은 초→밀리초 곱셈을 안정적으로 못 한다(10초를 4000으로 내는 식).
    시계와 산수는 코드가 맡는다는 원칙에 따라, 여기서 바로잡는다.
    자리 수가 일치할 때만 손대고, 애매하면 그대로 둔다.
    """
    wants = [int(x) * 1000 for x in re.findall(r"(\d+)\s*초", instruction)]
    if not wants:
        return commands
    slots = _duration_slots(commands)
    if not slots:
        return commands
    if len(slots) == len(wants):
        pairs = list(zip(slots, wants))            # 자리 수가 맞으면 순서대로 대응
    elif len(wants) == 1:
        pairs = [(max(slots, key=lambda s: s[1]), wants[0])]  # 가장 긴 대기가 본 대기
    else:
        return commands                            # 대응이 애매하면 건드리지 않는다
    for (path, got), want in pairs:
        if got != want:
            print(f"  ⏱ 지시는 {want//1000}초인데 계획은 {got/1000:g}초 → {want//1000}초로 맞춥니다")
            _set_duration(commands, path, want)
    return commands


def dataset_append(instruction: str, commands: list) -> int:
    """실행된 (지시 → 명령) 쌍을 JSONL로 기록. 누적 건수를 반환."""
    entry = {"instruction": instruction, "output": {"commands": commands}}
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def dataset_remove_last() -> int:
    """마지막 기록을 삭제(직전 동작이 잘못됐을 때). 남은 건수를 반환, 파일 없으면 -1."""
    if not os.path.exists(DATASET_FILE):
        return -1
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        entries = [line for line in f if line.strip()]
    if not entries:
        return -1
    entries.pop()
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        f.writelines(entries)
    return len(entries)


def _send_one(ser, line: str) -> bool:
    """명령 한 줄을 보내고 OK 응답을 기다린다. 성공 여부 반환."""
    print(f"  → {line}")
    ser.write((line + "\n").encode())
    deadline = time.time() + 20
    while time.time() < deadline:
        resp = ser.readline().decode(errors="ignore").strip()
        if not resp:
            continue
        if resp == "OK":
            return True
        if resp.startswith("ERR"):
            print(f"  ⚠ 아두이노 오류: {resp}")
            return True  # 오류지만 통신은 살아있으므로 계속
        print(f"  ← {resp}")  # POSE 등 정보성 출력
    print("  ⚠ 응답 시간 초과 — 연결/전원을 확인하세요")
    return False


def query_pose(ser):
    """아두이노에게 현재 각도를 물어본다. 실패 시 None."""
    ser.write(b"GET\n")
    pose = None
    deadline = time.time() + 5
    while time.time() < deadline:
        resp = ser.readline().decode(errors="ignore").strip()
        if not resp:
            continue
        if resp.startswith("POSE"):
            try:
                vals = [int(x) for x in resp.split()[1:]]
                pose = vals if len(vals) == NUM_JOINTS else None
            except ValueError:
                pose = None
        elif resp == "OK":
            return pose
        elif resp.startswith("ERR"):
            return None
    return None


def _exec(ser, item) -> bool:
    """명령 하나 실행. 상대 이동이면 현재 각도를 조회해 목표각을 계산한다."""
    if isinstance(item, dict) and "rel_joint" in item:
        pose = query_pose(ser)
        if pose is None:
            print("  ⚠ 현재 자세를 읽지 못해 상대 이동을 건너뜁니다")
            return True
        j = item["rel_joint"]
        lo, hi = JOINT_LIMITS[j]
        target = max(lo, min(hi, pose[j] + item["delta"]))
        print(f"  ↔ 관절 {j}: 현재 {pose[j]}도 {item['delta']:+d} → {target}도")
        return _send_one(ser, f"S {j} {target}")
    return _send_one(ser, item)


def send_serial(ser, lines: list):
    """명령 목록 실행. repeat 항목은 실제 시계를 보며 지정된 초 동안 반복한다."""
    for item in lines:
        if isinstance(item, dict) and "repeat_s" in item:
            secs = item["repeat_s"]
            print(f"  ⟳ {secs}초 동안 반복:")
            end = time.time() + secs
            while time.time() < end:
                cycle_start = time.time()
                for sub in item["lines"]:
                    if not _exec(ser, sub):
                        return
                    if time.time() >= end:
                        break
                if time.time() - cycle_start < 0.15:  # 지나치게 빠른 연사 방지
                    time.sleep(0.15)
        else:
            if not _exec(ser, item):
                return


def main():
    ap = argparse.ArgumentParser(description="Llama ↔ Arduino 로봇팔 브리지")
    ap.add_argument("--port", default="COM4", help="아두이노 시리얼 포트 (예: COM4)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 모델 이름")
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 LLM 변환만 테스트")
    args = ap.parse_args()

    ser = None
    if not args.dry_run:
        import serial  # pyserial
        ser = serial.Serial(args.port, 115200, timeout=0.5)
        time.sleep(2.5)  # 아두이노 리셋 대기
        ser.reset_input_buffer()
        print(f"[연결됨] {args.port} @115200")

    if not check_model(args.model):
        if ser:
            ser.close()
        return 1

    print(f"[모델] {args.model} — 자연어로 지시하세요. (종료: quit / 직전 기록 취소: bad / 직접 명령: !)")
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"[데이터셋] {existing}건 누적됨 → {DATASET_FILE}")
    history = []

    while True:
        try:
            user_text = input("\n명령> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "종료"):
            break
        if user_text.lower() == "stop":  # 비상: LLM 거치지 않고 즉시 홈
            if ser:
                send_serial(ser, ["HOME"])
            continue
        if user_text.startswith("!"):  # 직접 명령 (보정/디버깅용, LLM 안 거침)
            raw = user_text[1:].strip()
            if ser:
                send_serial(ser, [raw])
            else:
                print(f"[dry-run] {raw}")
            continue
        if user_text.lower() == "bad":  # 직전 동작이 이상했음 → 마지막 기록 삭제
            n = dataset_remove_last()
            if n < 0:
                print("삭제할 기록이 없습니다.")
            else:
                print(f"마지막 기록을 삭제했습니다. (데이터셋: {n}건)")
            continue

        try:
            commands = ask_llama(args.model, user_text, history)
            print(f"[Llama 계획] {json.dumps(commands, ensure_ascii=False)}")
            commands = enforce_duration(user_text, commands)
            commands = enforce_home_policy(user_text, commands)
            lines = validate(commands)
        except requests.ConnectionError:
            print("⚠ Ollama에 연결 실패 — 'ollama serve'가 실행 중인지 확인하세요.")
            continue
        except Exception as e:
            print(f"⚠ 변환/검증 실패: {e}")
            continue

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": json.dumps(commands, ensure_ascii=False)})

        if ser:
            send_serial(ser, lines)
            n = dataset_append(user_text, commands)
            print(f"[기록됨] 데이터셋 {n}건 (이상했다면 'bad' 입력으로 취소)")
        else:
            print("[dry-run] 전송 생략 (기록 안 함)")

    if ser:
        send_serial(ser, ["HOME"])
        ser.close()
    print("종료합니다.")


if __name__ == "__main__":
    sys.exit(main())
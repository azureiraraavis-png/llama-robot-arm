# -*- coding: utf-8 -*-
"""
colors.py — 한국어 색 이름 → HSV 범위 (순수 계산, 카메라와 무관)

"파란 물체 쪽을 봐" 에서 라마가 하는 일은 **"파란"이라는 낱말을 뽑는 것 하나뿐**입니다.
그 낱말을 실제 색 범위로 바꾸는 것은 여기, 물체를 찾는 것은 vision.py,
각도를 내는 것은 보정표가 합니다. 늘 하던 역할 분담입니다.

카메라가 없어도 시험할 수 있게 따로 뒀습니다.

OpenCV의 HSV는 H가 0~179입니다 (0~360이 아닙니다).

★ wrap 규약을 vision.make_mask 에 맞춥니다.
   make_mask는 h=[h0,h1]을 기본 구간으로 잡고, wrap=True면 **같은 폭**의 구간을
   반대쪽 끝(180-폭 ~ 180)에 하나 더 더합니다.
   그래서 빨강은 [170,10]이 아니라 **[0,9] + wrap** 으로 적어야 합니다.
   ([170,10]으로 적으면 lo>hi라 아무것도 안 잡힙니다. 실제로 그렇게 만들었다가 잡았습니다.)
"""

# (대표이름, 별칭들, H 범위, wrap)
TABLE = [
    # 빨강만 wrap=True — [0,9] 와 그 거울인 [171,180] 을 함께 잡습니다
    ("빨강", ("빨강", "빨간", "빨갛", "붉은", "적색", "레드", "red"), (0, 9), True),
    ("주황", ("주황", "오렌지", "귤색", "orange"), (10, 21), False),
    ("노랑", ("노랑", "노란", "노랗", "황색", "옐로", "yellow"), (22, 34), False),
    ("초록", ("초록", "녹색", "풀색", "그린", "green"), (38, 80), False),
    ("하늘", ("하늘", "청록", "민트", "시안", "cyan"), (82, 96), False),
    ("파랑", ("파랑", "파란", "파랗", "푸른", "청색", "블루", "blue"), (97, 124), False),
    ("보라", ("보라", "자주", "퍼플", "바이올렛", "purple"), (125, 148), False),
    ("분홍", ("분홍", "핑크", "연분홍", "pink"), (149, 170), False),
]

# 채도·명도의 기본 하한. 너무 낮추면 회색 배경이 잡히고, 너무 높이면 그늘에서 놓칩니다.
S_MIN, V_MIN = 80, 70

# 색을 말하지 않았을 때 — "보이는 것", "물체", "그거"
GENERIC = ("보이는", "보이", "물체", "그거", "그것", "저거", "저것", "대상", "타겟", "표적")


def make_range(h_lo, h_hi, wrap=False, s_min=S_MIN, v_min=V_MIN):
    return {"h": [h_lo, h_hi], "s": [s_min, 255], "v": [v_min, 255], "wrap": bool(wrap)}


def resolve(target, s_min=S_MIN, v_min=V_MIN):
    """색 이름이 들어간 말 → (대표이름, 범위). 색을 못 찾으면 (None, None).

    (None, None)은 실패가 아니라 "색을 지정하지 않았다"는 뜻입니다.
    부르는 쪽에서 저장된 설정(vision_config.json)을 쓰면 됩니다.
    """
    if not target:
        return None, None
    t = str(target).strip().lower()
    for name, aliases, (lo, hi), wrap in TABLE:
        for a in aliases:
            if a in t:
                return name, make_range(lo, hi, wrap, s_min, v_min)
    return None, None


def is_generic(target):
    """'보이는 것', '물체'처럼 색을 말하지 않은 표현인가."""
    if not target:
        return True
    t = str(target).strip().lower()
    return any(g in t for g in GENERIC)


def known_names():
    return [name for name, _a, _r, _w in TABLE]


def describe(target):
    """사람에게 보여줄 한 줄. 무엇을 찾을 것인지 분명히 말해 줍니다."""
    name, rng = resolve(target)
    if name:
        extra = " + 171~180" if rng["wrap"] else ""
        return f"{name}색 (H {rng['h'][0]}~{rng['h'][1]}{extra})"
    if is_generic(target):
        return "저장된 설정의 색 (vision_config.json)"
    return None


if __name__ == "__main__":
    print("\n  아는 색:", " ".join(known_names()))
    print("  색을 말하지 않으면 vision_config.json에 저장된 색을 씁니다.\n")
    for t in ("빨간 물체", "파란 것", "노란색 공", "보이는 것", "초록 블록",
              "핑크색", "그거", "무지개 물체"):
        name, rng = resolve(t)
        mark = "✔" if name else ("·" if is_generic(t) else "✘")
        print(f"  {mark} {t:<12} → {describe(t) or '모르는 색'}")
    print()
# -*- coding: utf-8 -*-
"""
hand_map.py — 손 랜드마크 → 팔 관절 각도 (순수 계산, 카메라·팔과 무관)

역할 분담을 지킨다. 손을 인식하는 것은 MediaPipe, 팔을 움직이는 것은 브리지,
그 사이의 '어떻게 대응시킬 것인가'만 이 파일이 맡는다. 그래서 카메라 없이도 시험할 수 있다.

MediaPipe 손 랜드마크 21점 (정규화 좌표 0~1)
   0 손목 · 4 엄지끝 · 8 검지끝 · 5 검지밑 · 9 중지밑 · 17 새끼밑

대응
  손의 좌우 위치      → 베이스 회전   (관절 0)
  손의 상하 위치      → 어깨         (관절 1)
  손의 크기(카메라 거리) → 팔꿈치       (관절 2)   가까이 = 팔을 뻗음
  손의 기울기         → 손목 회전     (관절 3)
  엄지-검지 벌림(핀치)  → 그리퍼       (관절 4)

거리에 따라 손 크기가 달라지므로, 핀치는 반드시 **손 폭으로 나눠** 정규화한다.
그러지 않으면 손을 멀리 뒀을 때 항상 '쥔 것'으로 읽힌다.
"""

import math

WRIST, THUMB_TIP, INDEX_TIP, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 4, 8, 5, 9, 17

# 팔이 연약하므로 하드웨어 한계보다 안쪽으로 (dance.py와 같은 방침)
SAFE = [(35, 145), (55, 125), (55, 125), (62, 118), (55, 115)]
HOME = [90, 90, 90, 90, 60]

# 손 크기(검지밑~새끼밑 거리)가 이 범위일 때 팔꿈치가 최대~최소로 움직인다.
# 화면 폭 대비 비율이며, 카메라와 20~50cm 거리에서 대략 이 정도가 나온다.
SPAN_NEAR, SPAN_FAR = 0.22, 0.08

# 핀치: 엄지끝~검지끝 거리 ÷ 손 폭. 완전히 붙이면 0.3 아래, 활짝 펴면 1.6 위.
PINCH_CLOSED, PINCH_OPEN = 0.45, 1.40


def _lerp(v, a, b, lo, hi):
    """v가 a~b 구간에서 어디쯤인지 보고 lo~hi로 옮긴다. 범위 밖은 끝값."""
    if a == b:
        return (lo + hi) / 2
    t = (v - a) / (b - a)
    return lo + max(0.0, min(1.0, t)) * (hi - lo)


def clamp(pose):
    return [int(max(lo, min(hi, round(v)))) for v, (lo, hi) in zip(pose, SAFE)]


def hand_to_pose(lm, mirror=True):
    """랜드마크 21점 → 관절 5개 각도.

    lm: [(x, y), ...] 또는 [(x, y, z), ...] 정규화 좌표 21개
    mirror: 카메라가 좌우 반전돼 보이므로 기본 True (거울처럼 움직인다)
    """
    if lm is None or len(lm) < 21:
        return None

    px = (lm[WRIST][0] + lm[INDEX_MCP][0] + lm[PINKY_MCP][0]) / 3     # 손바닥 중심
    py = (lm[WRIST][1] + lm[INDEX_MCP][1] + lm[PINKY_MCP][1]) / 3

    # ── 베이스: 손의 좌우 ────────────────────────────────────────────
    x = 1.0 - px if mirror else px
    base = _lerp(x, 0.15, 0.85, SAFE[0][0], SAFE[0][1])

    # ── 어깨: 손의 상하 (화면 위 = 팔을 세움) ─────────────────────────
    shoulder = _lerp(py, 0.15, 0.85, SAFE[1][1], SAFE[1][0])

    # ── 팔꿈치: 손 크기 = 카메라와의 거리 ─────────────────────────────
    span = math.dist(lm[INDEX_MCP][:2], lm[PINKY_MCP][:2])
    elbow = _lerp(span, SPAN_FAR, SPAN_NEAR, SAFE[2][1], SAFE[2][0])

    # ── 손목: 손의 기울기 (손목→중지밑 벡터의 각도) ───────────────────
    dx = lm[MIDDLE_MCP][0] - lm[WRIST][0]
    dy = lm[MIDDLE_MCP][1] - lm[WRIST][1]
    if mirror:
        dx = -dx
    tilt = math.degrees(math.atan2(dx, -dy))        # 손이 곧게 서면 0도
    wrist = _lerp(tilt, -60, 60, SAFE[3][0], SAFE[3][1])

    # ── 그리퍼: 핀치 (손 폭으로 정규화해 거리에 무관하게) ──────────────
    pinch = math.dist(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2]) / max(span, 1e-6)
    grip = _lerp(pinch, PINCH_CLOSED, PINCH_OPEN, SAFE[4][1], SAFE[4][0])

    return clamp([base, shoulder, elbow, wrist, grip])


class Smoother:
    """손은 떨리고 팔은 연약하다. 세 겹으로 거른다.

      1. 지수 평활(EMA)   — 프레임 단위의 잔떨림 제거
      2. 불감대           — 목표가 조금 바뀐 정도로는 움직이지 않음
      3. 최대 이동량 제한  — 한 번에 몇 도 이상 움직이지 않음 (급가속 방지)
    """

    def __init__(self, alpha=0.35, deadband=3, max_step=10):
        self.alpha = alpha
        self.deadband = deadband
        self.max_step = max_step
        self.smooth = None          # 평활된 목표
        self.sent = list(HOME)      # 실제로 팔에 보낸 값

    def update(self, target):
        """새 목표를 받아, 이번에 보낼 자세를 낸다. 보낼 필요 없으면 None."""
        if target is None:
            return None
        if self.smooth is None:
            self.smooth = list(target)
        else:
            a = self.alpha
            self.smooth = [s + a * (t - s) for s, t in zip(self.smooth, target)]

        if all(abs(s - c) < self.deadband for s, c in zip(self.smooth, self.sent)):
            return None                                    # 움직일 만큼 안 바뀜

        nxt = []
        for s, c in zip(self.smooth, self.sent):
            step = max(-self.max_step, min(self.max_step, s - c))
            nxt.append(c + step)
        self.sent = clamp(nxt)
        return list(self.sent)

    def reset_to(self, pose):
        self.sent = clamp(pose)
        self.smooth = None
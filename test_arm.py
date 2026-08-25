# -*- coding: utf-8 -*-
"""
test_arm.py — 클로드가 설계한 로봇팔 데모 시퀀스 (LLM 없이 하드웨어 검증)

실행:  python test_arm.py            (기본 COM4)
       python test_arm.py COM5       (다른 포트일 때)

동작: 홈 → 좌우 둘러보기 → 인사(손목 흔들기) → 그리퍼 테스트 → 홈 복귀
각 단계 사이에 관절이 잘 따라오는지 눈으로 확인하세요.
※ 시작 전 Arduino IDE의 시리얼 모니터는 반드시 닫아둘 것!
"""

import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM4"

# (설명, 명령) — robot_arm_llm.ino 프로토콜
SEQUENCE = [
    ("느린 속도로 설정",        "SPEED 3"),
    ("홈 자세",                "HOME"),
    ("오른쪽 둘러보기",         "S 0 40"),
    ("왼쪽 둘러보기",           "S 0 140"),
    ("정면 복귀",              "S 0 90"),
    ("살짝 앞으로 숙이기",      "S 1 70"),
    ("인사 1 — 손목 들기",      "S 3 60"),
    ("인사 2 — 손목 내리기",    "S 3 130"),
    ("인사 3 — 손목 들기",      "S 3 60"),
    ("손목 복귀",              "S 3 90"),
    ("그리퍼 열기",            "GRIP 0"),
    ("잠시 대기",              "WAIT 600"),
    ("그리퍼 닫기",            "GRIP 1"),
    ("그리퍼 열기",            "GRIP 0"),
    ("현재 자세 보고",          "GET"),
    ("홈 복귀",                "HOME"),
]


def main():
    print(f"[연결 시도] {PORT} @115200")
    ser = serial.Serial(PORT, 115200, timeout=0.5)
    time.sleep(2.5)  # 보드 리셋 대기
    ser.reset_input_buffer()
    print("[연결됨] 데모를 시작합니다. 팔 주변을 비워주세요!\n")
    time.sleep(1.0)

    for desc, cmd in SEQUENCE:
        print(f"▶ {desc:<16} ({cmd})")
        ser.write((cmd + "\n").encode())
        deadline = time.time() + 20
        while time.time() < deadline:
            resp = ser.readline().decode(errors="ignore").strip()
            if not resp:
                continue
            if resp == "OK":
                break
            if resp.startswith("ERR"):
                print(f"  ⚠ 아두이노 오류: {resp}")
                break
            print(f"  ← {resp}")
        else:
            print("  ⚠ 응답 없음 — 전원/펌웨어를 확인하세요")
            break
        time.sleep(0.3)

    ser.close()
    print("\n데모 완료! 모든 관절이 지시대로 움직였다면 Llama 연결 준비 끝입니다.")


if __name__ == "__main__":
    main()
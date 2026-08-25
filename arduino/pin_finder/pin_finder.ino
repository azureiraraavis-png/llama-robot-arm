/*
 * pin_finder.ino — 어떤 핀이 어떤 관절인지 확정하는 진단 스케치
 *
 * 업로드 후 시리얼 모니터(115200, "새 줄")에서:
 *   <핀번호> <각도>   예) 9 60   → D9 서보를 60도로
 *
 * 후보 핀 6개(D3, D5, D6, D9, D10, D11)를 전부 활성화해 둡니다.
 * 각 핀에 70 → 110을 차례로 보내 어느 부위가 움직이는지 기록하세요.
 * 예) "9 70" 입력 → 베이스가 움직였다 → D9=베이스 확정
 */

#include <Servo.h>

const uint8_t CANDIDATE_PINS[] = {3, 5, 6, 9, 10, 11};
const uint8_t NUM_PINS = sizeof(CANDIDATE_PINS);

Servo servos[NUM_PINS];

int pinIndex(long pin) {
  for (uint8_t i = 0; i < NUM_PINS; i++)
    if (CANDIDATE_PINS[i] == pin) return i;
  return -1;
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < NUM_PINS; i++) {
    servos[i].attach(CANDIDATE_PINS[i]);
    servos[i].write(90);
    delay(200);
  }
  Serial.println(F("PIN FINDER READY — 입력: <핀번호> <각도>  예) 9 60"));
}

void loop() {
  if (Serial.available()) {
    long pin = Serial.parseInt();
    long angle = Serial.parseInt();
    while (Serial.available()) Serial.read();  // 잔여 문자 비우기
    if (pin == 0 && angle == 0) return;

    int idx = pinIndex(pin);
    if (idx < 0) {
      Serial.print(F("ERR 후보 핀 아님: D"));
      Serial.println(pin);
      return;
    }
    if (angle < 40)  angle = 40;   // 진단 중 과도한 이동 방지
    if (angle > 140) angle = 140;
    servos[idx].write((int)angle);
    Serial.print(F("D"));
    Serial.print(pin);
    Serial.print(F(" → "));
    Serial.print(angle);
    Serial.println(F("도 (어느 부위가 움직였나요?)"));
  }
}

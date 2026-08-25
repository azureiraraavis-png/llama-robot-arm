/*
 * robot_arm_llm.ino
 * Adeept 5-DOF 로봇팔용 시리얼 명령 수신 펌웨어 (LLM 브리지용)
 *
 * PC(파이썬 + Ollama/Llama)에서 시리얼로 텍스트 명령을 보내면
 * 관절 한계를 강제하면서 서보를 부드럽게 움직입니다.
 *
 * ★ 중요: SERVO_PINS 배열을 실제 키트 배선(설명서 회로도)에 맞게 수정하세요.
 *   Adeept 5-DOF 키트(ADA031 등)는 보통 UNO R3의 PWM 핀에 서보를 연결합니다.
 *
 * 시리얼 프로토콜 (115200 baud, 한 줄에 한 명령, \n 종료):
 *   S <joint> <angle>          한 관절 이동   예) S 0 90
 *   P <a0> <a1> <a2> <a3> <a4> 전체 포즈 이동 예) P 90 60 120 90 40
 *   HOME                        홈 포즈로 이동
 *   GRIP <0|1>                  그리퍼 열기(0)/닫기(1)
 *   WAIT <ms>                   대기 (최대 5000ms)
 *   SPEED <1-10>                이동 속도 설정 (기본 5)
 *   GET                         현재 각도 출력
 * 응답: 성공 "OK", 오류 "ERR <사유>"
 *
 * 관절 번호: 0=베이스(회전), 1=어깨, 2=팔꿈치, 3=손목, 4=그리퍼
 */

#include <Servo.h>

const uint8_t NUM_JOINTS = 5;

// 실제 배선 (pin_finder로 실측 확정): 베이스=D9, 어깨=D6, 팔꿈치=D5, 손목=D3, 그리퍼=D11 (D10은 빈 포트)
const uint8_t SERVO_PINS[NUM_JOINTS] = {9, 6, 5, 3, 11};

// 관절별 안전 한계 [min, max] — 조립 후 실측해서 조정하세요.
// LLM이 어떤 값을 보내든 이 범위를 벗어나면 잘라냅니다(clamp).
const int JOINT_MIN[NUM_JOINTS] = {0,  15, 15, 10, 30};
const int JOINT_MAX[NUM_JOINTS] = {180, 165, 165, 170, 125};  // 그리퍼: 완전 닫힘 120(실측) + 여유 5

// 홈(초기) 포즈
const int HOME_POSE[NUM_JOINTS] = {90, 90, 90, 90, 60};

// 그리퍼 열림/닫힘 각도 (조립 후 실측 조정)
const int GRIP_OPEN  = 60;
const int GRIP_CLOSE = 120;  // 실측 보정값

Servo servos[NUM_JOINTS];
int currentAngle[NUM_JOINTS];
int speedLevel = 5;  // 1(느림)~10(빠름)

char lineBuf[64];
uint8_t lineLen = 0;

int clampAngle(uint8_t j, long a) {
  if (a < JOINT_MIN[j]) return JOINT_MIN[j];
  if (a > JOINT_MAX[j]) return JOINT_MAX[j];
  return (int)a;
}

// 모든 관절을 목표 각도로 "동시에, 부드럽게" 이동
void moveTo(const int target[NUM_JOINTS]) {
  int stepDelay = 22 - speedLevel * 2;  // speed 1→20ms, 10→2ms
  bool moving = true;
  while (moving) {
    moving = false;
    for (uint8_t j = 0; j < NUM_JOINTS; j++) {
      if (currentAngle[j] < target[j])      { currentAngle[j]++; moving = true; }
      else if (currentAngle[j] > target[j]) { currentAngle[j]--; moving = true; }
      servos[j].write(currentAngle[j]);
    }
    delay(stepDelay);
  }
}

void moveJoint(uint8_t j, int angle) {
  int target[NUM_JOINTS];
  for (uint8_t i = 0; i < NUM_JOINTS; i++) target[i] = currentAngle[i];
  target[j] = clampAngle(j, angle);
  moveTo(target);
}

void handleLine(char *line) {
  // 앞뒤 공백 제거는 생략(파이썬 쪽에서 정리해서 보냄)
  if (strncmp(line, "HOME", 4) == 0) {
    moveTo(HOME_POSE);
    Serial.println(F("OK"));
    return;
  }
  if (strncmp(line, "GET", 3) == 0) {
    Serial.print(F("POSE"));
    for (uint8_t j = 0; j < NUM_JOINTS; j++) { Serial.print(' '); Serial.print(currentAngle[j]); }
    Serial.println();
    Serial.println(F("OK"));
    return;
  }
  if (strncmp(line, "GRIP ", 5) == 0) {
    long v = atol(line + 5);
    moveJoint(4, v ? GRIP_CLOSE : GRIP_OPEN);
    Serial.println(F("OK"));
    return;
  }
  if (strncmp(line, "WAIT ", 5) == 0) {
    long ms = atol(line + 5);
    if (ms < 0) ms = 0;
    if (ms > 5000) ms = 5000;
    delay(ms);
    Serial.println(F("OK"));
    return;
  }
  if (strncmp(line, "SPEED ", 6) == 0) {
    long v = atol(line + 6);
    if (v < 1 || v > 10) { Serial.println(F("ERR speed 1-10")); return; }
    speedLevel = (int)v;
    Serial.println(F("OK"));
    return;
  }
  if (line[0] == 'S' && line[1] == ' ') {
    long j, a;
    if (sscanf(line + 2, "%ld %ld", &j, &a) == 2 && j >= 0 && j < NUM_JOINTS) {
      moveJoint((uint8_t)j, (int)a);
      Serial.println(F("OK"));
    } else {
      Serial.println(F("ERR bad S cmd"));
    }
    return;
  }
  if (line[0] == 'P' && line[1] == ' ') {
    long a[NUM_JOINTS];
    if (sscanf(line + 2, "%ld %ld %ld %ld %ld", &a[0], &a[1], &a[2], &a[3], &a[4]) == 5) {
      int target[NUM_JOINTS];
      for (uint8_t j = 0; j < NUM_JOINTS; j++) target[j] = clampAngle(j, a[j]);
      moveTo(target);
      Serial.println(F("OK"));
    } else {
      Serial.println(F("ERR bad P cmd"));
    }
    return;
  }
  Serial.println(F("ERR unknown cmd"));
}

void setup() {
  Serial.begin(115200);
  for (uint8_t j = 0; j < NUM_JOINTS; j++) {
    currentAngle[j] = HOME_POSE[j];
    servos[j].attach(SERVO_PINS[j]);
    servos[j].write(currentAngle[j]);
    delay(150);  // 전원 부하 분산을 위해 순차 기동
  }
  Serial.println(F("READY"));
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;  // 오버플로 시 라인 폐기
      Serial.println(F("ERR line too long"));
    }
  }
}

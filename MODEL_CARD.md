---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
language: [ko]
tags: [robotics, arduino, lora, qlora, korean]
---

# Llama-3.1-8B-arm-lora

**Built with Llama**

5축 아두이노 로봇팔을 한국어 지시로 조종하기 위한 QLoRA 어댑터입니다.
자연어 한 문장을 로봇팔 명령 JSON으로 옮깁니다.

> 이름이 `Llama`로 시작하는 것은 Llama 3.1 Community License 1.b의 요구사항입니다.
> 저장소 이름을 바꾸시더라도 이 접두어는 유지하세요.

## 하는 일

```
입력  "집게를 세 번 딱딱거리시오"
출력  {"commands": [{"cmd": "repeat", "times": 3,
                     "do": [{"cmd":"grip","close":true},
                            {"cmd":"grip","close":false}]}]}
```

## 학습

| | |
|---|---|
| 베이스 | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| 방식 | QLoRA, rank 16 |
| 학습 파라미터 | 41,943,040 / 8.07B (0.52%) |
| 데이터 | 98건 (한국어 지시 ↔ 명령 JSON), 손으로 11곳 교정 |
| 하드웨어 | RTX 4070 12GB |

파인튜닝의 목적은 성능보다 **프롬프트를 줄이는 것**이었습니다.
Ollama 판은 규칙과 예시가 든 긴 시스템 프롬프트를 쓰지만, 이 어댑터는 짧은 프롬프트로 같은 일을 합니다.

## 쓰는 법

이 어댑터만으로는 팔이 움직이지 않습니다. 명령을 검증하고 시리얼로 보내는 브리지가 필요합니다.

> 코드·펌웨어·데이터셋: *(여기에 GitHub 주소를 넣으세요)*

## 한계

- **한국어 지시 전용입니다.**
- **특정 개체에 맞춰져 있습니다** — 각도 한계와 그리퍼 닫힘각(120°)이 이 팔의 실측값입니다.
- 모델은 시간을 세지 않습니다. "n초"·"n번"은 브리지의 결정론적 코드가 처리합니다.
  그것이 설계 의도이며, 이 어댑터를 다른 시스템에 붙이면 그 보정이 없다는 뜻입니다.

## 라이선스

Llama 3.1 Community License. 베이스 모델은 Meta Llama 3.1 8B Instruct입니다.
https://www.llama.com/llama3_1/license/

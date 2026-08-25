# 라마와 로봇팔

**Llama 3.1 8B로 5축 아두이노 로봇팔을 자연어·손짓·목소리로 움직이기**

Adeept 5-DOF Robotic Arm Kit (Arduino UNO) · Llama 3.1 8B (Ollama + QLoRA) · RTX 4070 · Windows 11

---

> **In English —** A 5-DOF wooden robot arm (Arduino UNO) driven by a locally-run Llama 3.1 8B,
> in three ways: **plain Korean sentences**, **hand tracking** (MediaPipe → the arm mimics your hand),
> and **speech** (Whisper → the same pipeline). The language model only reads *intent*; all timing,
> arithmetic, repetition and safety limits are done by deterministic code, and the firmware clamps
> joint angles as a second layer.
>
> The most useful file here is probably not the code — it is
> [`대장정_기록.md`](대장정_기록.md), a build log (Korean) that records what went **wrong**:
> silkscreen that didn't match the wiring, contradictions in my own training data, an API that
> vanished between library versions, Windows Smart App Control blocking PyTorch, and the decision
> to abandon grasping for pointing when the hardware couldn't do it.

---

## 무엇을 하는가

```
"천천히 앞으로 숙여서 인사하시오"
"집게를 10번 딱딱거리시오"
"인사하고 나서, 30초 동안 로봇춤을 추시오"
"그 상태로 기다리세요"            ← 시간을 안 말하면 홈으로 안 돌아감
```

세 가지 입구가 있고, 그 뒤는 전부 같은 길입니다.

```
자연어 입력 ─┐
손 추적    ─┼→ 검증·보정 ─→ 시리얼 ─→ 아두이노 (각도 한계 강제)
음성 인식  ─┘
```

## 설계 원칙 — 이것이 거의 모든 문제를 풀었다

**언어모델에게는 언어 판단만 맡긴다.** 시계·산수·반복·안전 한계·의도 강제는 결정론적 코드가 한다.
아두이노는 각도 한계를 한 겹 더 강제한다.

모델이 "10초 기다려"를 4초로 잘못 옮겨도 `enforce_duration`이 지시문 쪽으로 되돌리고,
"그 상태로 유지하라"인데 홈 복귀를 붙여도 `home_policy`가 걷어낸다.
"n번 반복"은 모델이 세지 않는다 — `repeat` 명령만 내면 코드가 실제 시계를 보고 센다.

## 빠른 시작

```bash
# 1. 펌웨어 굽기
#    arduino/robot_arm_llm/robot_arm_llm.ino  (서보: D9 D6 D5 D3 D11)

# 2. 파이썬 (Windows, 3.12 권장 — torch가 3.12에만 설치됨)
py -3.12 -m pip install pyserial requests opencv-python "mediapipe==0.10.21"
#    torch/transformers는 CUDA 판으로 별도 설치

# 3. 라마 띄우기
ollama serve
ollama pull llama3.1:8b

# 4. 확인
py -3.12 checkup.py --deep        # 작업공간이 온전한지
py -3.12 mp_check.py              # mediapipe가 실제로 도는지
py -3.12 voice_check.py --stt     # 마이크·GPU·받아쓰기

# 5. 실행
py -3.12 llm_arm_bridge.py --port COM4    # 자연어 (Ollama)
py -3.12 arm_tuned.py                     # 자연어 (직접 구운 모델)
py -3.12 hand_track.py --dry-run          # 손 추적 (먼저 팔 없이)
py -3.12 voice_arm.py                     # 음성
```

## 프로그램

| 기능 | 파일 |
|---|---|
| **브리지** (검증·보정·전송의 단일 창구) | `llm_arm_bridge.py` |
| 학습된 모델로 조종 | `arm_tuned.py`, `arm_common.py`, `evaluate.py` |
| 춤 (안무 8종, 박자↔속도 역산) | `dance.py` |
| 카메라로 물체 보고 가리키기 | `vision.py`, `calibrate_point.py`, `point_at.py` |
| 손 추적 (MediaPipe → 5축) | `hand_map.py`, `hand_track.py` |
| 음성 (winmm 녹음 + Whisper) | `voice_io.py`, `stt.py`, `voice_arm.py` |
| 데이터 검사·정제 | `check_dataset.py`, `fix_dataset_v2.py` |
| 작업공간 이사·점검 | `move_project.py`, `checkup.py` |

**진단 도구가 이 저장소의 진짜 줄거리입니다.** 고비마다 답은 같았습니다 —
추측을 멈추고 재는 도구를 만드는 것.

`pin_finder.ino`(배선) · `check_dataset.py`(데이터) · `vision_check.py --diag`(카메라) ·
`mp_check.py`(mediapipe) · `voice_check.py`(마이크·GPU) · `checkup.py`(작업공간)

## 알려진 함정

이 프로젝트에서 실제로 하루씩 잡아먹은 것들입니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `WinError 4551` (mediapipe) | 0.10.30+ 가 서명 없는 `libmediapipe.dll`을 ctypes로 적재 | `mediapipe==0.10.21` (Tasks API 그대로 있음) |
| `WinError 4551` (torch) | 스마트 앱 제어가 `caffe2_nvrtc.dll` 차단. 개별 예외 없음 | Windows 보안 → 앱 및 브라우저 컨트롤 → 끄기 (24H2+는 되돌릴 수 있음) |
| `mp.solutions.hands` 없음 | 0.10.30에서 삭제됨. 인터넷 예제 대부분이 이 방식 | Tasks API + `hand_landmarker.task` |
| `No module named 'torch'` | `python`이 3.14로 감 | `py -3.12` |
| 카메라 "없음" | 윈도우 카메라 앱이 점유 중 | 앱을 닫기 (`vision_check.py --diag`로 확인) |
| 녹음이 8비트로 저장됨 | MCI 드라이버가 형식 요청을 무시 | 저장된 WAV 머리말을 읽고 변환 (`voice_io.load_wav`) |
| 화면 글자가 `???` | OpenCV 기본 글꼴이 한글을 못 그림 | 오버레이는 영어로 |

## 기록

**[`대장정_기록.md`](대장정_기록.md)** — 조립부터 음성까지 12장의 작업 기록.
잘 된 것보다 **틀린 것과 그걸 어떻게 알아냈는지**가 중심입니다.

보조 문서: [`가리키기_순서.md`](가리키기_순서.md) (카메라 보정 단계별 순서)

## 학습된 모델

QLoRA 어댑터 (rank 16, 41.9M 학습 파라미터 / 8.07B, 데이터 98건)는 용량 때문에
저장소에 넣지 않았습니다. Hugging Face Hub에 별도로 올려 두었습니다.

> **모델:** *https://huggingface.co/azureiraraavis/Llama-3.1-8B-arm-lora*

**Built with Llama** — 이 어댑터는 Meta Llama 3.1 8B Instruct를 파인튜닝한 파생물이며,
[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)를 따릅니다.

`arm_dataset_v3.jsonl`의 일부 출력은 Llama 3.1이 생성한 것을 사람이 교정한 것입니다.
이 데이터로 모델을 학습해 배포하실 경우, 그 모델 이름도 `Llama`로 시작해야 합니다.

직접 구우려면 데이터셋과 학습 스크립트가 저장소에 그대로 있습니다.

## 한계

솔직하게 적어 둡니다.

- **Windows 전용입니다.** 녹음이 `winmm.dll`, 키 감지가 `user32.dll`입니다. 의도적인 선택이었고
  (스마트 앱 제어에 막히지 않으려고) 그 대가로 이식성을 잃었습니다.
- **이 팔 하나에 맞춰져 있습니다.** 각도 한계, 그리퍼 닫힘각(120°), 보정값은 실측한 것이라
  다른 개체에는 그대로 맞지 않습니다. `*.json` 보정 파일은 예시로 보세요.
- **지시문이 한국어입니다.** 시스템 프롬프트와 데이터셋이 전부 한국어입니다.
- **역기구학이 없습니다.** 레이저 절단 나무와 마이크로 서보라 해석적 IK가 미덥지 않아,
  보정은 사람이 자세를 잡아 가르치는 방식(teach-by-demonstration)입니다.

## 라이선스

코드와 문서: MIT (`LICENSE` 참조 — 이름을 본인 것으로 바꾸세요)
학습된 어댑터: Llama 3.1 Community License

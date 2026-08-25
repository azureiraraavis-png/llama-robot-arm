# -*- coding: utf-8 -*-
"""
stt.py — 음성 → 글자 (Whisper, 이 컴퓨터의 GPU에서 직접)

왜 transformers인가: 이 컴퓨터에는 라마를 구울 때 쓴 torch와 transformers가
이미 깔려 있습니다. faster-whisper 같은 것을 새로 넣으면 또 서명 없는 DLL이
따라 들어와서 스마트 앱 제어에 막힐 여지가 생깁니다. 이미 도는 것으로 합니다.

인터넷은 처음 한 번, 모델을 내려받을 때만 씁니다. 그 뒤로는 전부 이 컴퓨터 안에서 돕니다.

모델 크기 (VRAM 12GB면 turbo가 넉넉합니다. 라마와 같이 띄워도 됩니다)
  openai/whisper-large-v3-turbo   약 1.6GB   한국어 정확도 좋음   ← 기본값
  openai/whisper-small            약 0.5GB   가볍지만 덜 정확
"""

import os
import sys
import time

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def wrong_python(missing):
    """3.14로 실행했을 때 나오는 안내. 진짜 원인은 '없다'가 아니라 '다른 파이썬'이다."""
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"{missing} 를 찾을 수 없습니다.\n"
        f"  지금 실행 중인 파이썬: {v}   ({sys.executable})\n"
        f"  torch와 transformers는 3.12에만 설치돼 있습니다.\n"
        f"  'python' 이 아니라 'py -3.12' 로 실행하세요:\n"
        f"      py -3.12 {os.path.basename(sys.argv[0]) or 'voice_arm.py'}"
    )


def python_ok():
    """torch가 이 파이썬에서 보이는가. (실제로 불러오지는 않는다 — 느리므로)"""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


class Whisper:
    def __init__(self, model_id=DEFAULT_MODEL, device=None, language="ko"):
        try:
            import torch
        except ImportError:
            raise RuntimeError(wrong_python("torch")) from None
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise RuntimeError(wrong_python("transformers")) from None

        self.torch = torch
        self.language = language
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        if device == "cpu":
            print("⚠ GPU를 찾지 못해 CPU로 돕니다. 많이 느립니다.")
        print(f"[음성인식] {model_id} 불러오는 중... (처음 한 번은 내려받느라 오래 걸립니다)")
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(model_id)
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=self.dtype)                 # transformers 5.x
        except TypeError:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.dtype)           # 그 이전 판
        self.model.to(device)
        self.model.eval()
        print(f"[음성인식] 준비 완료 ({time.time() - t0:.0f}초, {device})")

    def transcribe(self, audio, sample_rate=16000):
        """16kHz 모노 float32 배열 → 글자."""
        if audio is None or len(audio) < 800:      # 0.05초 미만
            return ""
        torch = self.torch
        inputs = self.proc(audio, sampling_rate=sample_rate, return_tensors="pt")
        feats = inputs.input_features.to(self.device, self.dtype)
        kw = {}
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            kw["attention_mask"] = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(feats, language=self.language,
                                      task="transcribe", max_new_tokens=96, **kw)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def cleanup(text: str) -> str:
    """Whisper가 붙이는 군더더기를 떼어냅니다.

    무음이나 잡음일 때 '시청해주셔서 감사합니다', 'MBC 뉴스 ...' 같은
    자막 상투구를 뱉는 버릇이 있습니다. 그대로 팔에 보내면 안 됩니다.
    """
    t = " ".join(text.split())
    junk = ("시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요",
            "감사합니다.", "MBC 뉴스", "KBS 뉴스", "Thanks for watching",
            "다음 영상에서 만나요", "한글자막 by", "字幕")
    for j in junk:
        if t.replace(" ", "") == j.replace(" ", ""):
            return ""
    return t


if __name__ == "__main__":
    # 파일 하나를 받아쓰기만 해 봅니다:  py -3.12 stt.py 녹음.wav
    import voice_io
    if len(sys.argv) < 2:
        print("사용법: py -3.12 stt.py <wav파일>")
        sys.exit(1)
    a, info = voice_io.load_wav(sys.argv[1])
    print(voice_io.describe(info))
    w = Whisper()
    print("→", repr(cleanup(w.transcribe(a))))
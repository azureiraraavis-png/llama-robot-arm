# -*- coding: utf-8 -*-
"""
voice_io.py — 마이크 녹음과 키 감지 (음성 인식과는 무관한 부분만)

설계 방침: **pip로 새 DLL을 들이지 않는다.**
지난번 mediapipe가 스마트 앱 제어에 막힌 것과 같은 일이 반복되기 쉬운데,
녹음 라이브러리(sounddevice, pyaudio)도 전부 서명 없는 DLL을 들고 다닙니다.
그래서 기본 녹음기는 윈도우에 원래 들어 있는 **winmm.dll**을 씁니다.
마이크로소프트가 서명한 시스템 파일이라 절대 막히지 않고, 설치할 것도 없습니다.
sounddevice가 이미 있으면 그쪽도 쓸 수 있게 남겨 뒀습니다.

녹음 형식은 MCI에게 16kHz·모노·16비트로 요청하지만, **믿지 않습니다.**
드라이버가 요청을 무시하는 일이 흔해서, 저장된 WAV의 머리말을 실제로 읽고
필요하면 여기서 변환합니다.
"""

import ctypes
import os
import wave

import numpy as np

try:
    import audioop                      # 파이썬 3.12까지 있음 (3.13에서 삭제)
except Exception:
    audioop = None

TARGET_RATE = 16000                     # Whisper가 원하는 값


# ── 키가 눌려 있는지 (누르고 있는 동안 녹음) ────────────────────────────

VK_SPACE, VK_ESCAPE, VK_RETURN = 0x20, 0x1B, 0x0D

try:
    _user32 = ctypes.WinDLL("user32.dll")
    _user32.GetAsyncKeyState.restype = ctypes.c_short
except Exception:
    _user32 = None


def key_down(vk=VK_SPACE):
    """지금 이 순간 그 키가 물리적으로 눌려 있는가."""
    if _user32 is None:
        return False
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def drain_keys():
    """콘솔에 쌓인 입력을 비웁니다. 녹음 중 눌린 스페이스가 뒤에 튀어나오지 않게."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass


# ── 녹음기 1: winmm (설치 불필요, 차단 없음) ────────────────────────────

class WinmmRecorder:
    name = "winmm"

    def __init__(self, rate=TARGET_RATE, alias="armmic"):
        self.dll = ctypes.WinDLL("winmm.dll")
        self.dll.mciSendStringW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                                            ctypes.c_uint, ctypes.c_void_p]
        self.dll.mciSendStringW.restype = ctypes.c_uint
        self.rate = rate
        self.alias = alias
        self.open = False

    def _cmd(self, s, soft=False):
        buf = ctypes.create_unicode_buffer(512)
        rc = self.dll.mciSendStringW(s, buf, 512, None)
        if rc:
            err = ctypes.create_unicode_buffer(512)
            self.dll.mciGetErrorStringW(rc, err, 512)
            if soft:
                return None                       # 드라이버가 거부해도 계속 진행
            raise RuntimeError(f"MCI {rc}: {err.value}  ← {s}")
        return buf.value

    def start(self):
        self.close_quietly()
        self._cmd(f"open new type waveaudio alias {self.alias}")
        self.open = True
        a, r = self.alias, self.rate
        # 아래 set들은 드라이버가 무시할 수 있습니다. 그래서 soft로 두고,
        # 실제 형식은 저장된 파일에서 확인합니다.
        for c in (f"set {a} time format ms",
                  f"set {a} format tag pcm",
                  f"set {a} bitspersample 16",
                  f"set {a} channels 1",
                  f"set {a} samplespersec {r}",
                  f"set {a} alignment 2",
                  f"set {a} bytespersec {r * 2}"):
            self._cmd(c, soft=True)
        self._cmd(f"record {self.alias}")

    def stop(self, path):
        self._cmd(f"stop {self.alias}", soft=True)
        path = os.path.abspath(path)
        if os.path.exists(path):
            os.remove(path)
        self._cmd(f'save {self.alias} "{path}"')
        self.close_quietly()
        return path

    def close_quietly(self):
        try:
            self._cmd(f"close {self.alias}", soft=True)
        except Exception:
            pass
        self.open = False


# ── 녹음기 2: sounddevice (이미 설치돼 있다면) ──────────────────────────

class SoundDeviceRecorder:
    name = "sounddevice"

    def __init__(self, rate=TARGET_RATE):
        import sounddevice as sd
        self.sd = sd
        self.rate = rate
        self.chunks = []
        self.stream = None

    def start(self):
        self.chunks = []

        def cb(indata, frames, t, status):
            self.chunks.append(indata.copy())

        self.stream = self.sd.InputStream(samplerate=self.rate, channels=1,
                                          dtype="int16", callback=cb)
        self.stream.start()

    def stop(self, path):
        self.stream.stop()
        self.stream.close()
        self.stream = None
        data = (np.concatenate(self.chunks) if self.chunks
                else np.zeros((0, 1), np.int16))
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(data.tobytes())
        return os.path.abspath(path)

    def close_quietly(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def make_recorder(prefer="auto"):
    """쓸 수 있는 녹음기를 만듭니다. 실패하면 (None, 이유들)."""
    errors = []
    order = {"auto": ["winmm", "sounddevice"],
             "winmm": ["winmm"], "sounddevice": ["sounddevice"]}[prefer]
    for which in order:
        try:
            rec = WinmmRecorder() if which == "winmm" else SoundDeviceRecorder()
            return rec, errors
        except Exception as e:
            errors.append((which, e))
    return None, errors


# ── 저장된 WAV → Whisper가 먹는 형태 ────────────────────────────────────

def _resample_numpy(a, src, dst):
    """audioop이 없을 때의 대체. 선형 보간."""
    if src == dst or len(a) == 0:
        return a
    n = int(round(len(a) * dst / float(src)))
    return np.interp(np.linspace(0, len(a) - 1, n),
                     np.arange(len(a)), a).astype(np.float32)


def load_wav(path):
    """WAV 하나를 읽어 (16kHz 모노 float32 배열, 원본 정보)로 돌려줍니다."""
    with wave.open(path, "rb") as w:
        nch, width, rate, n = (w.getnchannels(), w.getsampwidth(),
                               w.getframerate(), w.getnframes())
        raw = w.readframes(n)

    info = {"channels": nch, "bits": width * 8, "rate": rate,
            "seconds": (n / float(rate)) if rate else 0.0}

    if audioop is not None:
        if width == 1:
            # WAV의 8비트는 부호 없는 0~255인데 audioop은 부호 있는 값으로 봅니다.
            # 이걸 빠뜨리면 소리가 통째로 망가집니다 (드라이버가 8비트로 녹음하면 실제로 겪습니다).
            raw = audioop.bias(raw, 1, -128)
        if width != 2:
            raw = audioop.lin2lin(raw, width, 2)
        if nch > 1:
            raw = audioop.tomono(raw, 2, 0.5, 0.5)
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        if rate != TARGET_RATE and rate:
            raw2, _ = audioop.ratecv(
                (a * 32768.0).astype(np.int16).tobytes(), 2, 1, rate, TARGET_RATE, None)
            a = np.frombuffer(raw2, np.int16).astype(np.float32) / 32768.0
    else:
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dt is None:
            raise ValueError(f"다룰 수 없는 표본 크기: {width * 8}비트")
        a = np.frombuffer(raw, dt).astype(np.float32)
        a = (a - 128.0) / 128.0 if width == 1 else a / float(2 ** (width * 8 - 1))
        if nch > 1:
            a = a[:len(a) // nch * nch].reshape(-1, nch).mean(axis=1)
        a = _resample_numpy(a, rate, TARGET_RATE)

    info["level"] = float(np.abs(a).max()) if len(a) else 0.0
    info["rms"] = float(np.sqrt((a ** 2).mean())) if len(a) else 0.0
    return a.astype(np.float32), info


def describe(info):
    return (f"{info['seconds']:.1f}초 · {info['rate']}Hz · "
            f"{info['channels']}채널 · {info['bits']}비트 · "
            f"최대 {info['level']:.2f} / 평균 {info['rms']:.3f}")


def level_hint(info):
    """녹음이 실제로 들어왔는지 사람 말로."""
    if info["seconds"] < 0.2:
        return "⚠ 너무 짧습니다 — 키를 조금 더 오래 누르세요."
    if info["level"] < 0.005:
        return "⚠ 거의 무음입니다 — 마이크가 음소거이거나 다른 장치가 잡혔을 수 있습니다."
    if info["level"] < 0.05:
        return "△ 소리가 작습니다 — 마이크에 더 가까이, 또는 윈도우 마이크 볼륨을 올리세요."
    if info["level"] > 0.99:
        return "△ 소리가 너무 커서 잘렸습니다 — 조금 떨어져서 말해 보세요."
    return "✔ 녹음 상태 좋습니다."
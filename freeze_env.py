# -*- coding: utf-8 -*-
"""
freeze_env.py — 지금 잘 도는 이 컴퓨터의 환경을 그대로 받아 적습니다

pip freeze 를 그냥 쓰지 않는 이유: 이 컴퓨터에는 이 프로젝트와 무관한 패키지도
잔뜩 들어 있습니다. 전부 적어 두면 다음 컴퓨터에서 쓸데없는 것까지 깔다가
엉뚱한 데서 막힙니다. 그래서 **우리가 실제로 import 하는 것만** 골라 적습니다.

그리고 버전을 제 기억이나 문서에서 가져오지 않습니다.
지금 이 순간 실제로 설치된 것을 읽습니다. — 추측하지 말고 재는 것, 늘 하던 대로.

  py -3.12 freeze_env.py
"""

import json
import os
import platform
import subprocess
import sys

# 기능별로 무엇이 필요한가. (패키지 이름, import 이름)
TIERS = {
    "core": {
        "설명": "팔 조종 — 자연어 · 춤 · 아두이노 통신",
        "필요": [("pyserial", "serial"), ("requests", "requests")],
        "옵션": [],
    },
    "vision": {
        "설명": "카메라 · 가리키기 · 손 추적",
        "필요": [(("opencv-python", "opencv-contrib-python", "opencv-python-headless"), "cv2"),
                 ("mediapipe", "mediapipe"), ("numpy", "numpy")],
        "옵션": [],
    },
    "voice": {
        "설명": "음성 인식 (GPU 권장)",
        "필요": [("torch", "torch"), ("transformers", "transformers")],
        "옵션": [("accelerate", "accelerate"), ("safetensors", "safetensors")],
    },
    "train": {
        "설명": "모델 재학습 (평소에는 필요 없음)",
        "필요": [("peft", "peft"), ("datasets", "datasets")],
        "옵션": [("bitsandbytes", "bitsandbytes"), ("trl", "trl"), ("accelerate", "accelerate")],
    },
}

TORCH_INDEX = "https://download.pytorch.org/whl/"


def version_of(pkg, mod):
    """설치된 판을 읽습니다. 없으면 None.

    pkg 가 여러 개면(예: opencv-python / opencv-contrib-python) 실제로 깔린 쪽을 찾습니다.
    엉뚱한 이름으로 고정해 두면 다음 컴퓨터에서 두 판이 충돌합니다.
    """
    names = pkg if isinstance(pkg, tuple) else (pkg,)
    try:
        from importlib.metadata import version, PackageNotFoundError
        for name in names:
            try:
                return name, version(name)
            except PackageNotFoundError:
                continue
    except ImportError:
        pass
    try:                                   # 메타데이터가 없을 때의 대비책
        m = __import__(mod)
        v = getattr(m, "__version__", None)
        return (names[0], v) if v else (names[0], None)
    except Exception:
        return names[0], None


def torch_flavor():
    """torch가 CPU판인지 CUDA판인지. 이걸 놓치면 다음 컴퓨터에서 GPU를 못 씁니다."""
    try:
        import torch
    except Exception:
        return None, None
    v = torch.__version__
    cuda = getattr(torch.version, "cuda", None)
    tag = v.split("+")[1] if "+" in v else None      # 예: cu126
    return tag, cuda


def ollama_models():
    """Ollama에 무슨 모델이 올라가 있는지. pip로는 안 깔리는 의존성입니다."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def main():
    print(f"\n  파이썬 {sys.version.split()[0]}  ({sys.executable})")
    print(f"  {platform.system()} {platform.release()}\n")

    found, missing, written = {}, [], []

    for tier, spec in TIERS.items():
        lines = [f"# {tier} — {spec['설명']}",
                 f"# 이 컴퓨터에서 실제로 읽어 적은 값입니다 (freeze_env.py)",
                 ""]
        print(f"  [{tier}] {spec['설명']}")

        if tier == "voice":
            tag, cuda = torch_flavor()
            if tag:
                lines.insert(2, f"--extra-index-url {TORCH_INDEX}{tag}")
                lines.insert(3, "")
                print(f"    · torch 판: {tag} (CUDA {cuda})")
            else:
                print("    · torch가 CPU판이거나 없습니다")

        for pkg, mod in spec["필요"] + spec["옵션"]:
            pkg, v = version_of(pkg, mod)
            optional = any(p is mod for _p, p in spec["옵션"])
            optional = mod in [m for _p, m in spec["옵션"]]
            if v:
                found[pkg] = v
                lines.append(f"{pkg}=={v}")
                print(f"    ✔ {pkg:<18} {v}")
            elif optional:
                lines.append(f"# {pkg}  (이 컴퓨터에 없음 — 없어도 됩니다)")
                print(f"    · {pkg:<18} 없음 (선택)")
            else:
                missing.append((tier, pkg))
                lines.append(f"# {pkg}  ← 이 컴퓨터에 없어서 판을 적지 못했습니다")
                print(f"    ✘ {pkg:<18} 없음")

        path = f"requirements-{tier}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
        print()

    # ── pip로 안 깔리는 것들 ────────────────────────────────────────────
    tag, cuda = torch_flavor()
    report = {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "torch_flavor": tag,
        "cuda": cuda,
        "packages": found,
        "ollama_models": ollama_models(),
    }
    try:
        import torch
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass

    with open("env_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    written.append("env_report.json")

    print("  " + "─" * 56)
    print("  적어 둔 파일")
    for w in written:
        print(f"    {w}")

    print("\n  pip로는 옮겨지지 않는 것들 (수동 확인 필요)")
    print(f"    · 파이썬 {report['python']}  — 3.12여야 합니다")
    if report.get("gpu"):
        print(f"    · GPU {report['gpu']} ({report['vram_gb']}GB)")
    if report.get("ollama_models"):
        print(f"    · Ollama 모델 {', '.join(report['ollama_models'])}")
    else:
        print("    · Ollama — 지금 꺼져 있어 확인하지 못했습니다")
    print("    · 아두이노 펌웨어, 그리고 보정값 (*.json) — 이 팔 한 개체의 실측값")

    if missing:
        print("\n  ⚠ 판을 적지 못한 것")
        for tier, pkg in missing:
            print(f"    {tier}: {pkg}")
        print("    이 컴퓨터에 없는 것이니, 쓰지 않는 기능이면 그대로 두셔도 됩니다.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
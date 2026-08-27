# -*- coding: utf-8 -*-
"""
setup_env.py — 새 컴퓨터에 이 프로젝트 환경을 세웁니다

파이썬으로 쓴 이유: PowerShell 스크립트는 실행 정책에 막힙니다(실제로 겪었습니다).
파이썬은 어차피 깔아야 하니, 설치 도구도 파이썬이 확실합니다.

가상환경(venv)을 만드는 이유: `python`과 `py -3.12`가 서로 다른 파이썬을 가리켜서
"torch가 없다"는 엉뚱한 오류를 겪었습니다. venv를 쓰면 그 혼동이 사라집니다.
폴더 안의 .venv 하나만 보면 되니까요.

  py -3.12 setup_env.py                기본(core+vision) 설치
  py -3.12 setup_env.py --all          음성·학습까지 전부
  py -3.12 setup_env.py --tier core    팔 조종만
  py -3.12 setup_env.py --check        무엇이 설치될지만 보기
  py -3.12 setup_env.py --system-site  이미 깔린 torch를 재사용 (내려받기 절약)
"""

import argparse
import os
import subprocess
import sys
import venv

TIERS = ["core", "vision", "voice", "train"]
DEFAULT = ["core", "vision"]
VENV = ".venv"


def venv_python(root=VENV):
    if os.name == "nt":
        return os.path.join(root, "Scripts", "python.exe")
    return os.path.join(root, "bin", "python")


def run(cmd, quiet=False):
    print(f"    $ {' '.join(str(c) for c in cmd[:4])}{' ...' if len(cmd) > 4 else ''}")
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    return r.returncode == 0, (r.stderr or "") if quiet else ""


def check_python():
    v = sys.version_info
    print(f"  파이썬 {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) == (3, 12):
        print("    ✔ 3.12 — 맞습니다")
        return True
    print(f"    ✘ 3.12가 아닙니다.")
    print("      torch의 CUDA 판이 3.12까지만 나와 있어서 3.12여야 합니다.")
    print("      python.org 에서 3.12를 받아 설치한 뒤 'py -3.12 setup_env.py' 로 실행하세요.")
    return False


def main():
    ap = argparse.ArgumentParser(description="환경 설치")
    ap.add_argument("--tier", action="append", choices=TIERS,
                    help="설치할 묶음 (여러 번 쓸 수 있음). 기본: core vision")
    ap.add_argument("--all", action="store_true", help="네 묶음 전부")
    ap.add_argument("--check", action="store_true", help="무엇이 설치될지만 보기")
    ap.add_argument("--system-site", action="store_true",
                    help="이미 깔린 패키지를 venv에서 재사용 (torch 재다운로드 방지)")
    ap.add_argument("--no-venv", action="store_true",
                    help="venv 없이 지금 파이썬에 바로 설치")
    args = ap.parse_args()

    tiers = TIERS if args.all else (args.tier or DEFAULT)

    print()
    if not check_python():
        return 1

    # ── 어떤 목록이 있는지 확인 ────────────────────────────────────────
    print("\n  설치할 묶음")
    plan = []
    for t in tiers:
        path = f"requirements-{t}.txt"
        if not os.path.exists(path):
            print(f"    ✘ {path} 가 없습니다")
            print("      원래 컴퓨터에서 'py -3.12 freeze_env.py' 를 먼저 돌리세요.")
            return 1
        with open(path, encoding="utf-8") as f:
            pkgs = [l.strip() for l in f
                    if l.strip() and not l.startswith("#") and not l.startswith("-")]
        plan.append((t, path, pkgs))
        print(f"    [{t}] {len(pkgs)}개  —  {', '.join(p.split('==')[0] for p in pkgs)}")

    if any(t == "voice" for t, _p, _k in plan):
        print("\n    ※ voice 묶음은 torch를 받습니다. 2GB 이상이고 시간이 꽤 걸립니다.")
        print("      이미 torch가 깔려 있다면 --system-site 를 붙이세요.")

    if args.check:
        print("\n  확인까지입니다. 실제로 설치하려면 --check 없이 다시 실행하세요.\n")
        return 0

    # ── venv ─────────────────────────────────────────────────────────
    if args.no_venv:
        py = sys.executable
        print(f"\n  venv 없이 설치합니다 → {py}")
    else:
        print(f"\n  가상환경 만들기 → {VENV}{'  (기존 패키지 재사용)' if args.system_site else ''}")
        if os.path.exists(venv_python()):
            print("    이미 있습니다. 그대로 씁니다.")
        else:
            venv.EnvBuilder(with_pip=True,
                            system_site_packages=args.system_site).create(VENV)
            print("    ✔ 만들었습니다")
        py = venv_python()

    ok, _ = run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # ── 설치 ─────────────────────────────────────────────────────────
    print("\n  설치합니다")
    failed = []
    for t, path, _pkgs in plan:
        print(f"  [{t}]")
        ok, err = run([py, "-m", "pip", "install", "-r", path], quiet=False)
        if not ok:
            failed.append(t)
            print(f"    ✘ {t} 설치 실패")

    # ── 확인 ─────────────────────────────────────────────────────────
    print("\n  확인합니다")
    if os.path.exists("checkup.py"):
        run([py, "checkup.py", "--deep"])
    else:
        print("    checkup.py 가 없어 건너뜁니다")

    print("\n" + "  " + "─" * 56)
    if failed:
        print(f"  실패한 묶음: {', '.join(failed)}")
        print("  위의 pip 오류를 보시고, 필요하면 해당 requirements 파일의 판을 낮춰 보세요.")
    else:
        print("  설치가 끝났습니다.")

    if not args.no_venv:
        run_py = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" \
                 else os.path.join(VENV, "bin", "python")
        print("\n  앞으로는 이렇게 실행하세요 (py -3.12 대신)")
        print(f"    {run_py} llm_arm_bridge.py --port COM4")
        print(f"    {run_py} hand_track.py --dry-run")
        print("\n  또는 한 번 활성화해 두고 그냥 python 으로 쓰셔도 됩니다")
        print(f"    {os.path.join(VENV, 'Scripts', 'Activate.ps1')}" if os.name == "nt"
              else f"    source {os.path.join(VENV, 'bin', 'activate')}")

    print("\n  pip로는 옮겨지지 않는 것 — 손으로 확인하세요")
    print("    1. Ollama 설치 후:  ollama pull llama3.1:8b")
    print("    2. 아두이노에 robot_arm_llm.ino 굽기 (서보 D9 D6 D5 D3 D11)")
    print("    3. 보정값(*.json)은 팔마다 다릅니다 — 새 팔이면 다시 재야 합니다")
    print("    4. NVIDIA 드라이버 (음성·재학습을 쓸 경우)\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
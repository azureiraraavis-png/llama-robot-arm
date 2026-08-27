# -*- coding: utf-8 -*-
"""
publish_model.py — 학습된 어댑터를 Hugging Face Hub에 올립니다

왜 GitHub이 아니라 여기냐면: git은 큰 파일에 약하고, 지운 뒤에도 모든 판을 영원히 보관합니다.
Hugging Face는 모델을 두라고 만든 곳이라 큰 파일 처리가 이미 들어 있습니다.

준비
  py -3.12 -m pip install huggingface_hub      (순수 파이썬 — 새 DLL이 없어 차단될 일 없음)
  (로그인은 이 스크립트가 필요할 때 직접 물어봅니다 — CLI 명령이 없어도 됩니다)

실행
  py -3.12 publish_model.py --user 내아이디
  py -3.12 publish_model.py --user 내아이디 --dry-run    무엇이 올라갈지만 보기
  py -3.12 publish_model.py --user 내아이디 --card-only  모델 카드만 다시 올리기
  py -3.12 publish_model.py --user 내아이디 --verify     올린 것이 제대로 있는지 확인

이름이 Llama 로 시작하는 것은 취향이 아니라 라이선스 요구사항입니다.
  Llama 3.1 Community License 1.b — 파생 모델 이름은 "Llama"로 시작해야 합니다.
"""

import argparse
import getpass
import json
import os
import sys

DEFAULT_NAME = "Llama-3.1-8B-arm-lora"
CARD = "MODEL_CARD.md"


def human(n):
    if n >= 1024 * 1024:
        return f"{n/1024/1024:.1f}MB"
    return f"{n/1024:.0f}KB" if n >= 1024 else f"{n}B"


def find_adapter(hint=None):
    """adapter_config.json 이 들어 있는 폴더를 찾습니다. 이름은 무엇이든 상관없습니다."""
    if hint:
        return hint if os.path.exists(os.path.join(hint, "adapter_config.json")) else None
    for name in sorted(os.listdir(".")):
        if os.path.isdir(name) and os.path.exists(os.path.join(name, "adapter_config.json")):
            return name
    return None


def survey(folder):
    """올릴 것과 건너뛸 것을 갈라 봅니다."""
    send, skip = [], []
    for cur, dirs, files in os.walk(folder):
        rel_dir = os.path.relpath(cur, folder)
        first = rel_dir.replace("\\", "/").split("/")[0]
        target = skip if first.startswith("checkpoint-") else send
        for f in files:
            p = os.path.join(cur, f)
            target.append((os.path.relpath(p, folder), os.path.getsize(p)))
    return sorted(send), sorted(skip)


def check_base(folder):
    """어댑터가 어느 베이스 모델을 가리키는지. 카드와 어긋나면 남이 못 씁니다."""
    try:
        with open(os.path.join(folder, "adapter_config.json"), encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path")
    except Exception:
        return None


def ensure_login(token=None):
    """로그인 상태를 확인하고, 안 돼 있으면 여기서 바로 받습니다.

    huggingface-cli(또는 hf) 명령이 PATH에 없는 일이 흔해서 — 파이썬 패키지는 깔려도
    명령어 파일은 Scripts 폴더에 따로 놓이고 그 폴더가 PATH에 없는 경우가 많습니다 —
    CLI에 기대지 않고 이 안에서 처리합니다.
    """
    from huggingface_hub import HfApi, login

    api = HfApi()
    try:
        return api.whoami()["name"]              # 이미 로그인돼 있음
    except Exception:
        pass

    if token is None:
        print("\n로그인이 필요합니다.")
        print("  1. https://huggingface.co/settings/tokens 에서 토큰을 만드세요")
        print("     ★ 권한(Role)은 반드시 Write 여야 합니다. Read로는 못 올립니다.")
        print("  2. 만든 토큰(hf_... 로 시작)을 아래에 붙여넣으세요.")
        print("     PowerShell에서는 마우스 오른쪽 클릭이 붙여넣기입니다.")
        print("     보안을 위해 입력해도 화면에는 아무것도 안 보입니다. 정상입니다.\n")
        try:
            token = getpass.getpass("  토큰> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n취소했습니다.")
            return None
    if not token:
        print("  토큰이 비어 있습니다.")
        return None
    if not token.startswith("hf_"):
        print("  ⚠ 보통 토큰은 'hf_' 로 시작합니다. 잘못 붙여넣지 않았는지 확인하세요.")

    try:
        login(token=token, add_to_git_credential=False)
        who = HfApi().whoami()["name"]
    except Exception as e:
        print(f"  ⚠ 로그인 실패 — {type(e).__name__}: {e}")
        print("    권한이 Write인지, 토큰을 통째로 붙여넣었는지 확인하세요.")
        return None

    print(f"  ✔ {who} 으로 로그인했습니다. (다음부터는 안 묻습니다)")
    return who


def verify(repo_id, folder):
    """올린 것이 실제로 그쪽에 있는지 확인합니다. 눈으로 보는 것보다 확실합니다."""
    from huggingface_hub import HfApi, hf_hub_download

    url = f"https://huggingface.co/{repo_id}"
    print(f"\n확인합니다: {url}\n")
    try:
        remote = set(HfApi().list_repo_files(repo_id))
    except Exception as e:
        print(f"  ✘ 저장소를 읽지 못했습니다 — {type(e).__name__}: {e}")
        print("    주소(아이디·이름)가 맞는지, 비공개라면 로그인했는지 확인하세요.")
        return 1

    bad = 0

    # 1. 있어야 할 파일이 다 있는가
    if folder:
        send, _skip = survey(folder)
        missing = [f for f, _n in send if f.replace("\\", "/") not in remote]
        if missing:
            bad += 1
            print(f"  ✘ 빠진 파일 {len(missing)}개: {', '.join(missing[:4])}")
        else:
            print(f"  ✔ 어댑터 파일 {len(send)}개 전부 올라가 있습니다")

    # 2. 안 올라갔어야 할 것이 올라갔는가
    ckpt = [f for f in remote if f.replace("\\", "/").startswith("checkpoint-")]
    if ckpt:
        bad += 1
        print(f"  ✘ 학습 중간 저장본이 올라갔습니다 ({len(ckpt)}개) — 지우는 게 좋습니다")
    else:
        print("  ✔ 학습 중간 저장본은 올라가지 않았습니다")

    # 3. 모델 카드가 제구실을 하는가
    if "README.md" not in remote:
        print("  ✘ README.md(모델 카드)가 없습니다")
        return 1
    try:
        text = open(hf_hub_download(repo_id, "README.md"), encoding="utf-8").read()
    except Exception as e:
        print(f"  ✘ 모델 카드를 읽지 못했습니다 — {e}")
        return 1

    for ok, label, hint in (
        ("Built with Llama" in text, "'Built with Llama' 표시",
         "라이선스가 요구하는 문구입니다"),
        ("llama.com/llama3_1/license" in text, "라이선스 링크", ""),
        ("여기에 GitHub 주소를 넣으세요" not in text, "GitHub 링크가 채워짐",
         "--card-only 로 카드를 다시 올리세요"),
        ("github.com" in text, "GitHub 주소가 들어 있음", ""),
    ):
        if ok:
            print(f"  ✔ {label}")
        else:
            bad += 1
            print(f"  ✘ {label}" + (f" — {hint}" if hint else ""))

    print()
    if bad:
        print(f"  {bad}가지가 아직입니다.\n")
        return 1
    print(f"  전부 제자리입니다. → {url}\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description="어댑터를 Hugging Face에 올립니다")
    ap.add_argument("--user", required=True, help="Hugging Face 아이디")
    ap.add_argument("--name", default=DEFAULT_NAME, help=f"모델 이름 (기본 {DEFAULT_NAME})")
    ap.add_argument("--folder", default=None, help="어댑터 폴더 (기본: 자동으로 찾음)")
    ap.add_argument("--private", action="store_true", help="비공개로 만들기")
    ap.add_argument("--dry-run", action="store_true", help="무엇이 올라갈지만 보기")
    ap.add_argument("--card-only", action="store_true", help="모델 카드만 다시 올리기")
    ap.add_argument("--verify", action="store_true",
                    help="올린 것이 실제로 그쪽에 있는지 확인만 하기")
    ap.add_argument("--token", default=None,
                    help="Hugging Face 토큰 (생략하면 필요할 때 물어봅니다)")
    args = ap.parse_args()

    if not args.name.startswith("Llama"):
        print(f"⚠ 모델 이름이 'Llama'로 시작해야 합니다. (지금: {args.name})")
        print("  Llama 3.1 Community License 1.b 의 요구사항입니다.")
        return 1

    repo_id = f"{args.user}/{args.name}"
    url = f"https://huggingface.co/{repo_id}"

    folder = find_adapter(args.folder)

    if args.verify:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("⚠ huggingface_hub 가 없습니다:  py -3.12 -m pip install huggingface_hub")
            return 1
        return verify(repo_id, folder)

    if folder is None and not args.card_only:
        print("⚠ adapter_config.json 이 있는 폴더를 찾지 못했습니다.")
        print("  작업공간 폴더에서 실행하고 있는지 확인하세요. (--folder 로 지정도 가능)")
        return 1

    if not os.path.exists(CARD):
        print(f"⚠ {CARD} 가 없습니다. 이 폴더에서 실행하세요.")
        return 1

    print(f"\n  올릴 곳: {url}")
    if folder:
        print(f"  어댑터 : {folder}")
        base = check_base(folder)
        if base:
            print(f"  베이스 : {base}")

        send, skip = survey(folder)
        total = sum(n for _f, n in send)
        print(f"\n올릴 파일 {len(send)}개 ({human(total)})")
        for f, n in send:
            print(f"      {f:<40} {human(n):>9}")
        if skip:
            print(f"\n  건너뜀: 학습 중간 저장본 {len(skip)}개 "
                  f"({human(sum(n for _f, n in skip))}) — 학습 재개용이라 배포에는 불필요")
    print(f"\n  모델 카드: {CARD} → 저장소의 README.md 로 올라갑니다")

    if args.dry_run:
        print("\n미리보기까지입니다. 실제로 올리려면 --dry-run 없이 다시 실행하세요.\n")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\n⚠ huggingface_hub 가 없습니다:")
        print("    py -3.12 -m pip install huggingface_hub")
        return 1

    who = ensure_login(args.token)
    if who is None:
        return 1
    api = HfApi()
    if who != args.user:
        print(f"\n  참고: 로그인 계정은 '{who}' 인데 --user 는 '{args.user}' 입니다.")
        print("  조직 계정이 아니라면 --user 를 고치세요.")

    print("\n올립니다...")
    try:
        api.create_repo(repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        if not args.card_only:
            api.upload_folder(folder_path=folder, repo_id=repo_id,
                              ignore_patterns=["checkpoint-*", "*.lock", "runs/*"])
        api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md", repo_id=repo_id)
    except Exception as e:
        print(f"⚠ 실패 — {type(e).__name__}: {e}")
        return 1

    print(f"\n  ✔ 완료 → {url}")
    print("\n다음 (링크 두 곳을 서로 채워 넣기)")
    print(f"  1. README.md 에서    (여기에 Hugging Face 주소를 넣으세요)")
    print(f"     →  {url}")
    print(f"  2. {CARD} 에서   (여기에 GitHub 주소를 넣으세요)")
    print(f"     →  https://github.com/<GitHub아이디>/llama-robot-arm")
    print(f"  3. git commit & push")
    print(f"  4. py -3.12 publish_model.py --user {args.user} --card-only")
    print("     ← 카드를 고쳤으니 한 번 더 올려야 Hugging Face 쪽에도 반영됩니다")
    print(f"  5. py -3.12 publish_model.py --user {args.user} --verify")
    print("     ← 전부 제자리인지 확인\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
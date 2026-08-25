# -*- coding: utf-8 -*-
"""
move_project.py — 로봇팔 프로젝트를 새 작업공간으로 옮깁니다

기본은 '복사'입니다. 원본은 건드리지 않습니다.
새 자리에서 전부 잘 도는 것을 확인한 뒤에 원본을 직접 지우세요.

  py -3.12 move_project.py --dry-run     ← 먼저 이걸로 무엇이 옮겨질지만 봅니다
  py -3.12 move_project.py               ← 실제로 복사

  --src / --dst 로 위치를 바꿀 수 있습니다.

파일 이름을 외워서 적지 않았습니다. 대신 표시를 보고 찾습니다.
  · 학습된 모델 폴더 → adapter_config.json 이 들어 있는 폴더
  · 아두이노 스케치  → .ino 가 들어 있는 폴더
그래서 제가 이름을 잘못 기억하고 있어도 빠뜨리지 않습니다.
"""

import argparse
import os
import shutil
import sys

EXTS = (".py", ".json", ".jsonl", ".md", ".task", ".ino")
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".venv"}
SKIP_NAMES = {"move_project.py", "move_project.ps1"}     # 이사 도구 자신
KIND = {".py": "프로그램", ".json": "설정", ".jsonl": "데이터",
        ".md": "문서", ".task": "인식 모델", ".ino": "아두이노"}

DEFAULT_DST = r"D:\workspace_raraavis\arduino_project00"


def human(n):
    return f"{n/1024/1024:.1f}MB" if n >= 1024 * 1024 else f"{n/1024:.0f}KB"


def walk(root, max_depth):
    """max_depth 단계까지만 훑습니다. 쓰레기 폴더는 들어가지 않습니다."""
    root = os.path.abspath(root)
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if d not in SKIP_DIRS]
        yield cur, files


def is_checkpoint(rel_path):
    """학습 중간 저장본인가. 학습을 이어서 할 때만 쓰이고, 실행에는 필요 없습니다."""
    first = rel_path.replace("\\", "/").split("/")[0]
    return first.startswith("checkpoint-")


def build_plan(src, dst, with_checkpoints=False):
    """(원본, 대상, 종류) 목록과, 건너뛴 중간 저장본 정보를 만듭니다."""
    plan, claimed = [], set()
    skipped = {"count": 0, "bytes": 0}

    def add(frm, to, kind):
        if frm in claimed:
            return
        claimed.add(frm)
        plan.append((frm, to, kind))

    # 1. 학습된 모델 폴더 — adapter_config.json 이 있는 곳을 통째로
    for cur, files in walk(src, 3):
        if "adapter_config.json" in files:
            name = os.path.basename(cur)
            for sub, subfiles in walk(cur, 5):
                rel_dir = os.path.relpath(sub, cur)
                for f in subfiles:
                    rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                    if not with_checkpoints and is_checkpoint(rel):
                        skipped["count"] += 1
                        skipped["bytes"] += os.path.getsize(os.path.join(sub, f))
                        continue
                    add(os.path.join(sub, f), os.path.join(dst, name, rel), "학습된 모델")

    # 2. 아두이노 스케치 — arduino\<이름>\ 으로 정리
    for cur, files in walk(src, 3):
        for f in files:
            if not f.lower().endswith(".ino"):
                continue
            name = os.path.splitext(f)[0]
            if os.path.basename(cur) == name:
                # 이미 제 이름의 폴더 안에 있음 → 같이 있는 .h 등도 함께
                for g in files:
                    add(os.path.join(cur, g),
                        os.path.join(dst, "arduino", name, g), "아두이노")
            else:
                add(os.path.join(cur, f),
                    os.path.join(dst, "arduino", name, f), "아두이노")

    # 3. 프로그램·설정·데이터·문서 — 원본 폴더 바로 아래만
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if not os.path.isfile(p) or f in SKIP_NAMES:
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in EXTS:
            add(p, os.path.join(dst, f), KIND[ext])

    return plan, claimed, skipped


def show(plan, dst):
    print(f"\n옮길 것 ({len(plan)}개)\n")
    by_kind = {}
    for frm, to, kind in plan:
        by_kind.setdefault(kind, []).append((frm, to))
    for kind in sorted(by_kind):
        items = by_kind[kind]
        total = sum(os.path.getsize(f) for f, _ in items)
        print(f"  {kind:<10} {len(items):>3}개  {human(total):>9}")
        if len(items) <= 30:
            for _f, t in sorted(items, key=lambda x: x[1]):
                print(f"      {os.path.relpath(t, dst)}")     # 새 자리에서의 경로
        else:
            print(f"      ... {len(items)}개 (많아서 생략)")
        print()


def main():
    ap = argparse.ArgumentParser(description="작업공간 이사")
    ap.add_argument("--src", default=os.path.join(os.path.expanduser("~"), "Downloads"))
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--dry-run", action="store_true", help="무엇이 옮겨질지만 봅니다")
    ap.add_argument("--with-checkpoints", action="store_true",
                    help="학습 중간 저장본(checkpoint-*)도 함께 가져옵니다")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    print(f"\n  원본: {src}")
    print(f"  대상: {dst}")
    if args.dry_run:
        print("  (미리보기 — 아무것도 복사하지 않습니다)")

    if not os.path.isdir(src):
        print(f"\n원본 폴더가 없습니다: {src}")
        return 1
    if os.path.abspath(dst).startswith(src + os.sep):
        print("\n대상이 원본 안에 있습니다. 다른 곳을 고르세요.")
        return 1
    drive = os.path.splitdrive(dst)[0]
    if drive and not os.path.exists(drive + os.sep):
        print(f"\n{drive} 드라이브를 찾을 수 없습니다. 연결돼 있는지 확인하세요.")
        return 1

    plan, claimed, skipped = build_plan(src, dst, args.with_checkpoints)
    if not plan:
        print("\n옮길 것을 찾지 못했습니다. --src 를 확인하세요.")
        return 1
    show(plan, dst)
    if skipped["count"]:
        print(f"  건너뜀: 학습 중간 저장본 {skipped['count']}개 ({human(skipped['bytes'])})")
        print("    학습을 이어서 할 때만 쓰이고, 팔을 움직이는 데는 필요 없습니다.")
        print("    원본에 그대로 남아 있습니다. 함께 가져오려면 --with-checkpoints\n")

    if args.dry_run:
        print("미리보기까지입니다. 실제로 복사하려면 --dry-run 없이 다시 실행하세요.\n")
        return 0

    print("복사합니다...")
    failed = []
    for frm, to, _kind in plan:
        try:
            os.makedirs(os.path.dirname(to), exist_ok=True)
            shutil.copy2(frm, to)
        except Exception as e:
            failed.append((frm, f"{type(e).__name__}: {e}"))

    print("확인합니다...")
    bad = []
    for frm, to, _kind in plan:
        if not os.path.exists(to):
            bad.append(f"없음: {to}")
        elif os.path.getsize(frm) != os.path.getsize(to):
            bad.append(f"크기 다름: {to}")

    if not failed and not bad:
        print(f"  ✔ {len(plan)}개 전부 크기까지 일치합니다.")
    else:
        for f, why in failed:
            print(f"  ✘ 복사 실패: {f} — {why}")
        for b in bad:
            print(f"  ✘ {b}")

    leftover = [f for f in sorted(os.listdir(src))
                if os.path.isfile(os.path.join(src, f))
                and os.path.splitext(f)[1].lower() in EXTS
                and os.path.join(src, f) not in claimed
                and f not in SKIP_NAMES]
    if leftover:
        print("\n원본에 남은 같은 종류 파일 (필요하면 직접 옮기세요):")
        for f in leftover:
            print(f"      {f}")

    print("\n다음 순서")
    print(f'  cd "{dst}"')
    print("  py -3.12 checkup.py --deep")
    print("\n원본은 그대로 남아 있습니다. 확인이 끝난 뒤에 직접 지우세요.\n")
    return 1 if (failed or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
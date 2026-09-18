#!/usr/bin/env python3
"""رابط خط فرمان ArshivJooy — بدون نیاز به Gradio.

نمونه‌ها::

    python cli.py models                       # دانلود/بررسی مدل‌ها
    python cli.py index ~/Archive              # ایندکس کردن پوشه
    python cli.py index ~/Archive --force      # ایندکس مجدد کامل
    python cli.py search "قرارداد اجاره" -n 10
    python cli.py search "گربه روی مبل" --image ~/sample.jpg
    python cli.py stats
    python cli.py reset --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import TYPE_DOCUMENT, TYPE_IMAGE, get_settings
from src.ingest import ingest_folder
from src.models import ensure_all_models, is_downloaded
from src.search import get_search_engine
from src.store import get_store


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# دستورات
# --------------------------------------------------------------------------- #

def cmd_models(args: argparse.Namespace) -> int:
    settings = get_settings()
    for label, repo_id in (("متن", settings.text_model_id), ("عکس", settings.image_model_id)):
        state = "موجود" if is_downloaded(repo_id, settings) else "دانلود نشده"
        print(f"[{label}] {repo_id} — {state}")

    print("\nدر حال آماده‌سازی مدل‌ها …")
    models = ensure_all_models(settings)
    for kind, model in models.items():
        print(f"  ✓ {kind}: {model.path}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"خطا: پوشه پیدا نشد → {folder}", file=sys.stderr)
        return 1

    stats = ingest_folder(folder, force=args.force, show_progress=not args.quiet)
    print()
    print(stats.summary_fa())
    return 0 if stats.failed == 0 else 2


def cmd_search(args: argparse.Namespace) -> int:
    type_filter = None
    if args.only == "image":
        type_filter = TYPE_IMAGE
    elif args.only == "document":
        type_filter = TYPE_DOCUMENT

    results = get_search_engine().search(
        args.query,
        image_path=args.image,
        n_results=args.n_results,
        type_filter=type_filter,
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("نتیجه‌ای پیدا نشد.")
        return 0

    for position, result in enumerate(results, start=1):
        kind = "عکس" if result.type == TYPE_IMAGE else "سند"
        print(f"\n{position}. {result.filename}  [{kind}]  امتیاز: {result.score:.3f}")
        print(f"   مسیر: {result.path}")
        print(f"   منابع تطبیق: {', '.join(result.matched_by)}")
        if result.text_snippet:
            print(f"   متن: {result.text_snippet}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    settings = get_settings()
    stats = get_store().stats()
    print(f"مسیر دیتابیس: {settings.chroma_dir}")
    print(f"فایل‌های ایندکس‌شده: {stats['files']}")
    print(f"تکه‌های متنی: {stats['text_chunks']}")
    print(f"بردارهای تصویری: {stats['image_vectors']}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        answer = input("کل ایندکس پاک شود؟ (y/N) ").strip().lower()
        if answer not in {"y", "yes"}:
            print("لغو شد.")
            return 1
    get_store().reset()
    print("ایندکس پاک شد.")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arshivjooy",
        description="موتور جست‌وجوی معنایی محلی برای آرشیو شخصی (عکس + سند)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="نمایش لاگ کامل")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_models = subparsers.add_parser("models", help="دانلود یا بررسی مدل‌ها")
    p_models.set_defaults(func=cmd_models)

    p_index = subparsers.add_parser("index", help="ایندکس‌کردن یک پوشه")
    p_index.add_argument("folder", help="مسیر پوشه‌ی آرشیو")
    p_index.add_argument("--force", action="store_true", help="ایندکس مجدد حتی اگر تغییری نکرده باشد")
    p_index.add_argument("--quiet", action="store_true", help="بدون نوار پیشرفت")
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="جست‌وجو در آرشیو")
    p_search.add_argument("query", nargs="?", default="", help="پرس‌وجوی متنی")
    p_search.add_argument("-n", "--n-results", type=int, default=5, help="تعداد نتایج")
    p_search.add_argument("--image", help="مسیر عکس نمونه برای جست‌وجوی ترکیبی")
    p_search.add_argument(
        "--only",
        choices=["image", "document", "all"],
        default="all",
        help="محدودکردن نوع نتایج",
    )
    p_search.add_argument("--json", action="store_true", help="خروجی JSON")
    p_search.set_defaults(func=cmd_search)

    p_stats = subparsers.add_parser("stats", help="آمار ایندکس")
    p_stats.set_defaults(func=cmd_stats)

    p_reset = subparsers.add_parser("reset", help="پاک‌کردن کامل ایندکس")
    p_reset.add_argument("--yes", action="store_true", help="بدون پرسش تأیید")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def _force_utf8_stdio() -> None:
    # Windows consoles default to a legacy codepage (e.g. cp1256) that can't encode Persian.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "search" and not args.query and not args.image:
        parser.error("برای جست‌وجو حداقل یک پرس‌وجوی متنی یا --image لازم است.")

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nمتوقف شد.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"خطا: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

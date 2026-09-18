#!/usr/bin/env python3
"""رابط گرافیکی فارسی ArshivJooy (Gradio).

اجرا::

    python app.py
"""

from __future__ import annotations

import base64
import html
import io
import logging
from pathlib import Path

import gradio as gr

from src.config import TYPE_DOCUMENT, TYPE_IMAGE, get_settings
from src.ingest import ingest_folder
from src.models import ensure_all_models, is_downloaded
from src.search import SearchResult, get_search_engine
from src.store import get_store

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = (180, 180)

CUSTOM_CSS = """
.gradio-container { direction: rtl; font-family: Vazirmatn, Tahoma, "IRANSans", sans-serif; }
.gradio-container textarea, .gradio-container input[type="text"] { direction: rtl; text-align: right; }
#aj-header { text-align: center; padding: 8px 0 2px; }
#aj-header h1 { margin: 0; font-size: 1.9rem; }
#aj-header p { margin: 4px 0 0; opacity: .75; }
.aj-card {
    display: flex; gap: 14px; align-items: flex-start;
    border: 1px solid var(--border-color-primary, #e3e3e3);
    border-radius: 12px; padding: 14px; margin-bottom: 12px;
    background: var(--background-fill-secondary, #fafafa);
}
.aj-card img { width: 110px; height: 110px; object-fit: cover; border-radius: 8px; flex-shrink: 0; }
.aj-card .aj-body { flex: 1; min-width: 0; }
.aj-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; word-break: break-all; }
.aj-path { font-size: .8rem; opacity: .65; direction: ltr; text-align: left; word-break: break-all; margin-bottom: 8px; }
.aj-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.aj-badge { font-size: .75rem; padding: 2px 9px; border-radius: 999px; background: #e8eefc; color: #23408e; }
.aj-badge.score { background: #e6f6ec; color: #1c6b3a; }
.aj-snippet { font-size: .92rem; line-height: 1.9; white-space: pre-wrap; }
.aj-empty { text-align: center; padding: 28px; opacity: .7; }
"""


# --------------------------------------------------------------------------- #
# کمکی‌ها
# --------------------------------------------------------------------------- #

def _thumbnail_data_uri(path: str) -> str | None:
    """تولید بندانگشتی base64 برای نمایش داخل کارت."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


_SOURCE_LABELS = {
    "text": "متن",
    "clip_text": "مفهوم تصویری",
    "clip_image": "شباهت عکس",
}


def _render_card(index: int, result: SearchResult) -> str:
    kind = "عکس" if result.type == TYPE_IMAGE else "سند"
    badges = [
        f'<span class="aj-badge score">امتیاز {result.score:.3f}</span>',
        f'<span class="aj-badge">{kind}</span>',
    ]
    badges += [
        f'<span class="aj-badge">{_SOURCE_LABELS.get(source, source)}</span>'
        for source in dict.fromkeys(result.matched_by)
    ]

    thumbnail = _thumbnail_data_uri(result.path) if result.type == TYPE_IMAGE else None
    image_html = f'<img src="{thumbnail}" alt="" />' if thumbnail else ""

    return f"""
    <div class="aj-card">
      {image_html}
      <div class="aj-body">
        <div class="aj-title">{index}. {html.escape(result.filename)}</div>
        <div class="aj-path">{html.escape(result.path)}</div>
        <div class="aj-badges">{''.join(badges)}</div>
        <div class="aj-snippet">{html.escape(result.text_snippet)}</div>
      </div>
    </div>
    """


def _render_results(results: list[SearchResult]) -> str:
    if not results:
        return '<div class="aj-empty">نتیجه‌ای پیدا نشد. ابتدا پوشه را ایندکس کنید یا عبارت دیگری را امتحان کنید.</div>'
    return "".join(_render_card(i, result) for i, result in enumerate(results, start=1))


def _index_status() -> str:
    try:
        stats = get_store().stats()
    except Exception as exc:
        return f"وضعیت ایندکس نامشخص: {exc}"
    return (
        f"فایل‌های ایندکس‌شده: {stats['files']} | "
        f"تکه‌های متنی: {stats['text_chunks']} | "
        f"بردارهای تصویری: {stats['image_vectors']}"
    )


# --------------------------------------------------------------------------- #
# اکشن‌ها
# --------------------------------------------------------------------------- #

def action_prepare_models(progress: gr.Progress = gr.Progress()) -> str:
    settings = get_settings()
    lines = []
    for label, repo_id in (("متنی", settings.text_model_id), ("تصویری", settings.image_model_id)):
        if is_downloaded(repo_id, settings):
            lines.append(f"✓ مدل {label} از قبل موجود است.")
        else:
            lines.append(f"⬇ در حال دانلود مدل {label} …")
    progress(0.1, desc="آماده‌سازی مدل‌ها")
    try:
        ensure_all_models(settings)
    except Exception as exc:
        return "\n".join(lines + [f"✗ خطا: {exc}"])
    progress(1.0, desc="آماده")
    return "\n".join(lines + ["✓ هر دو مدل آماده‌اند؛ از این پس اجرا کاملاً آفلاین است."])


def action_index(
    folder: str,
    force: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    folder = (folder or "").strip()
    if not folder:
        return "لطفاً مسیر پوشه را وارد کنید.", _index_status()

    path = Path(folder).expanduser()
    if not path.exists():
        return f"پوشه پیدا نشد: {path}", _index_status()

    progress(0.0, desc="در حال آماده‌سازی مدل‌ها …")

    def on_progress(current: int, total: int, filename: str) -> None:
        progress(current / max(total, 1), desc=f"({current}/{total}) {filename}")

    try:
        stats = ingest_folder(
            path,
            force=force,
            show_progress=True,          # نوار tqdm در ترمینال
            progress_callback=on_progress,  # نوار Gradio در مرورگر
        )
    except Exception as exc:
        logger.exception("ایندکس ناموفق بود")
        return f"خطا در ایندکس: {exc}", _index_status()

    return stats.summary_fa(), _index_status()


def action_search(
    query: str,
    image_path: str | None,
    n_results: int,
    scope: str,
) -> str:
    query = (query or "").strip()
    if not query and not image_path:
        return '<div class="aj-empty">یک عبارت فارسی بنویسید یا یک عکس نمونه بارگذاری کنید.</div>'

    type_filter = {"فقط عکس": TYPE_IMAGE, "فقط سند": TYPE_DOCUMENT}.get(scope)

    try:
        results = get_search_engine().search(
            query,
            image_path=image_path,
            n_results=int(n_results),
            type_filter=type_filter,
        )
    except Exception as exc:
        logger.exception("جست‌وجو ناموفق بود")
        return f'<div class="aj-empty">خطا در جست‌وجو: {html.escape(str(exc))}</div>'

    return _render_results(results)


# --------------------------------------------------------------------------- #
# رابط
# --------------------------------------------------------------------------- #

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="آرشیوجو", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            '<div id="aj-header"><h1>آرشیوجو</h1>'
            "<p>جست‌وجوی معنایی در آرشیو شخصی — کاملاً محلی و آفلاین</p></div>"
        )

        with gr.Tab("جست‌وجو"):
            with gr.Row():
                query_box = gr.Textbox(
                    label="عبارت جست‌وجو",
                    placeholder="مثلاً: قرارداد اجاره سال ۱۴۰۲ … یا: عکس ساحل در غروب",
                    lines=2,
                    scale=4,
                )
                sample_image = gr.Image(
                    label="عکس نمونه (اختیاری)",
                    type="filepath",
                    height=150,
                    scale=1,
                )
            with gr.Row():
                n_results = gr.Slider(1, 30, value=5, step=1, label="تعداد نتایج")
                scope = gr.Radio(
                    ["همه", "فقط عکس", "فقط سند"],
                    value="همه",
                    label="نوع نتایج",
                )
            search_button = gr.Button("جست‌وجو", variant="primary")
            results_html = gr.HTML(
                '<div class="aj-empty">هنوز جست‌وجویی انجام نشده است.</div>',
                label="نتایج",
            )

            search_button.click(
                action_search,
                inputs=[query_box, sample_image, n_results, scope],
                outputs=results_html,
            )
            query_box.submit(
                action_search,
                inputs=[query_box, sample_image, n_results, scope],
                outputs=results_html,
            )

        with gr.Tab("ایندکس کردن پوشه"):
            folder_box = gr.Textbox(
                label="مسیر پوشه‌ی آرشیو",
                placeholder="/home/user/Archive",
            )
            force_checkbox = gr.Checkbox(
                label="ایندکس مجدد همه‌ی فایل‌ها (نادیده‌گرفتن حافظه‌ی قبلی)",
                value=False,
            )
            with gr.Row():
                index_button = gr.Button("ایندکس کردن پوشه", variant="primary")
                models_button = gr.Button("دانلود / بررسی مدل‌ها")

            index_log = gr.Textbox(label="گزارش", lines=12, interactive=False)
            status_box = gr.Textbox(label="وضعیت ایندکس", interactive=False, value=_index_status)

            index_button.click(
                action_index,
                inputs=[folder_box, force_checkbox],
                outputs=[index_log, status_box],
            )
            models_button.click(action_prepare_models, outputs=index_log)

        gr.Markdown(
            "پسوندهای پشتیبانی‌شده: `.pdf` `.txt` `.docx` `.jpg` `.jpeg` `.png` — "
            "متنِ عکس‌ها با OCR (فارسی + انگلیسی) استخراج می‌شود."
        )

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_interface().launch(server_name="127.0.0.1", inbrowser=True, show_api=False)


if __name__ == "__main__":
    main()

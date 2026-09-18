"""استخراج متن از اسناد و عکس‌ها (OCR)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Iterable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_NEWLINES_RE = re.compile(r"\n{3,}")

# نگاشت اعداد و نویسه‌های عربی به معادل فارسی/استاندارد
_NORMALIZATION_MAP = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "‌": " ",  # نیم‌فاصله
        "‏": "",   # RLM
        "‎": "",   # LRM
        "﻿": "",
    }
)


class ExtractionError(RuntimeError):
    """استخراج متن از فایل ناموفق بود."""


def normalize_text(text: str) -> str:
    """یکسان‌سازی نویسه‌های فارسی/عربی و فشرده‌کردن فاصله‌ها."""
    if not text:
        return ""
    text = text.translate(_NORMALIZATION_MAP)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _AROUND_NEWLINE_RE.sub("\n", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# اسناد
# --------------------------------------------------------------------------

def extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError(f"رمزگذاری فایل متنی قابل تشخیص نیست: {path}")


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"خواندن PDF ناموفق بود: {path}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # صفحه‌ی خراب نباید کل فایل را از کار بیندازد
            logger.warning("استخراج صفحه %d از %s ناموفق بود", index + 1, path.name)
    return "\n\n".join(pages)


def extract_docx(path: Path) -> str:
    from docx import Document

    try:
        document = Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"خواندن DOCX ناموفق بود: {path}") from exc

    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


_DOCUMENT_EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".txt": extract_txt,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
}


def extract_document_text(path: Path) -> str:
    """متن یک سند را بر اساس پسوند آن استخراج می‌کند."""
    extractor = _DOCUMENT_EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ExtractionError(f"پسوند پشتیبانی‌نشده: {path.suffix}")
    return normalize_text(extractor(path))


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

_tesseract_configured = False


def _configure_tesseract(settings: Settings) -> None:
    global _tesseract_configured
    if _tesseract_configured:
        return
    import pytesseract

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    _tesseract_configured = True


def ocr_image(path: Path, settings: Settings | None = None) -> str:
    """متن داخل عکس را با Tesseract استخراج می‌کند.

    اگر Tesseract نصب نباشد یا زبان فارسی موجود نباشد، به‌جای خطا
    رشته‌ی خالی برمی‌گرداند تا ایندکسِ برداریِ تصویر ادامه یابد.
    """
    settings = settings or get_settings()
    try:
        import pytesseract
        from PIL import Image

        _configure_tesseract(settings)
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image.convert("RGB"), lang=settings.ocr_languages)
        return normalize_text(text)
    except Exception as exc:
        logger.warning("OCR روی %s ناموفق بود (%s)", path.name, exc)
        return ""


# --------------------------------------------------------------------------
# تکه‌بندی
# --------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int, max_chunks: int) -> list[str]:
    """متن را به تکه‌های هم‌پوشان تقسیم می‌کند (مرزها روی فاصله)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    overlap = max(0, min(overlap, chunk_size // 2))
    step = chunk_size - overlap
    length = len(text)
    chunks: list[str] = []
    start = 0

    while start < length and len(chunks) < max_chunks:
        end = min(start + chunk_size, length)
        if end < length:
            # مرز را روی نزدیک‌ترین فاصله بعد از حداقلِ گامِ لازم قرار می‌دهیم
            boundary = text.rfind(" ", start + step, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def iter_supported_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    """همه‌ی فایل‌های پشتیبانی‌شده‌ی زیرِ یک پوشه را (بازگشتی) برمی‌گرداند."""
    allowed = {ext.lower() for ext in extensions}
    if not root.exists():
        raise FileNotFoundError(f"پوشه پیدا نشد: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in allowed else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)

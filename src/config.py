"""پیکربندی مرکزی پروژه ArshivJooy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# --- شناسه مدل‌ها روی Hugging Face ---
TEXT_MODEL_ID: Final[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
IMAGE_MODEL_ID: Final[str] = "openai/clip-vit-base-patch32"

# --- نام کالکشن‌ها در ChromaDB ---
# کالکشن اصلی: بردار متنی (۳۸۴ بعدی) برای اسناد و متنِ OCR عکس‌ها.
TEXT_COLLECTION: Final[str] = "archive"
# کالکشن کمکی: بردار تصویری CLIP (۵۱۲ بعدی) — ابعاد متفاوت است و
# ChromaDB اجازه‌ی نگه‌داری دو بُعد مختلف در یک کالکشن را نمی‌دهد.
IMAGE_COLLECTION: Final[str] = "archive_images"

DOCUMENT_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf", ".txt", ".docx"})
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png"})
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS

ItemType = str  # "document" | "image"
TYPE_DOCUMENT: Final[ItemType] = "document"
TYPE_IMAGE: Final[ItemType] = "image"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """تنظیمات قابل‌تغییر از طریق متغیرهای محیطی."""

    data_dir: Path = field(default_factory=lambda: _env_path("ARSHIVJOOY_DATA_DIR", PROJECT_ROOT / "data"))
    text_model_id: str = TEXT_MODEL_ID
    image_model_id: str = IMAGE_MODEL_ID

    # OCR
    ocr_languages: str = field(default_factory=lambda: os.getenv("ARSHIVJOOY_OCR_LANGS", "fas+eng"))
    tesseract_cmd: str | None = field(default_factory=lambda: os.getenv("ARSHIVJOOY_TESSERACT_CMD"))

    # تکه‌بندی متن (مدل متنی پنجره‌ی کوتاهی دارد)
    chunk_size: int = field(default_factory=lambda: _env_int("ARSHIVJOOY_CHUNK_SIZE", 480))
    chunk_overlap: int = field(default_factory=lambda: _env_int("ARSHIVJOOY_CHUNK_OVERLAP", 80))
    max_chunks_per_file: int = field(default_factory=lambda: _env_int("ARSHIVJOOY_MAX_CHUNKS", 400))

    # طول متنی که داخل metadata ذخیره می‌شود
    metadata_text_limit: int = 1200

    # دستگاه محاسباتی: "auto" | "cpu" | "cuda"
    device: str = field(default_factory=lambda: os.getenv("ARSHIVJOOY_DEVICE", "auto"))

    embedding_batch_size: int = field(default_factory=lambda: _env_int("ARSHIVJOOY_BATCH_SIZE", 32))

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """نمونه‌ی یکتا (singleton) از تنظیمات."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
        _SETTINGS.ensure_dirs()
    return _SETTINGS

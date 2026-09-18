"""مدیریت دانلود و کش محلی مدل‌ها.

در اولین اجرا وزن‌ها از Hugging Face دانلود و در ``data/models`` ذخیره می‌شوند؛
از آن پس کتابخانه‌ها در حالت offline اجرا می‌شوند و هیچ درخواست شبکه‌ای
فرستاده نمی‌شود.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# فقط فایل‌های لازم؛ از دانلود نسخه‌های tf/flax/onnx جلوگیری می‌کند.
_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
    "openvino/*",
    "*.tflite",
)

_MARKER_NAME = ".arshivjooy_ready"


class ModelDownloadError(RuntimeError):
    """دانلود مدل ناموفق بود و نسخه‌ی محلی هم وجود ندارد."""


@dataclass(frozen=True, slots=True)
class LocalModel:
    """مسیر محلی یک مدل آماده به استفاده."""

    repo_id: str
    path: Path

    def __str__(self) -> str:  # pragma: no cover - نمایشی
        return f"{self.repo_id} -> {self.path}"


def _local_dir(settings: Settings, repo_id: str) -> Path:
    return settings.models_dir / repo_id.replace("/", "__")


def is_downloaded(repo_id: str, settings: Settings | None = None) -> bool:
    """آیا مدل قبلاً به‌طور کامل دانلود شده است؟"""
    settings = settings or get_settings()
    return (_local_dir(settings, repo_id) / _MARKER_NAME).exists()


def ensure_model(repo_id: str, settings: Settings | None = None) -> LocalModel:
    """در صورت نبودِ مدل آن را دانلود می‌کند و مسیر محلی را برمی‌گرداند."""
    settings = settings or get_settings()
    target = _local_dir(settings, repo_id)

    if (target / _MARKER_NAME).exists():
        return LocalModel(repo_id, target)

    logger.info("در حال دانلود مدل %s ...", repo_id)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - وابستگی نصب‌نشده
        raise ModelDownloadError("huggingface-hub نصب نیست: pip install -r requirements.txt") from exc

    # اگر کاربر قبلاً حالت offline را روشن کرده باشد، دانلود ممکن نیست.
    offline_flags = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
    previous = {k: os.environ.pop(k, None) for k in offline_flags}
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            ignore_patterns=list(_IGNORE_PATTERNS),
        )
    except Exception as exc:
        raise ModelDownloadError(
            f"دانلود مدل «{repo_id}» ناموفق بود. برای اولین اجرا به اینترنت نیاز است.\nجزئیات: {exc}"
        ) from exc
    finally:
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value

    (target / _MARKER_NAME).write_text("ok", encoding="utf-8")
    logger.info("مدل %s آماده شد.", repo_id)
    return LocalModel(repo_id, target)


def ensure_all_models(settings: Settings | None = None) -> dict[str, LocalModel]:
    """هر دو مدل متنی و تصویری را آماده می‌کند."""
    settings = settings or get_settings()
    return {
        "text": ensure_model(settings.text_model_id, settings),
        "image": ensure_model(settings.image_model_id, settings),
    }


def enable_offline_mode() -> None:
    """کتابخانه‌های HF را وادار می‌کند فقط از کش محلی بخوانند."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

"""تولید بردار (embedding) برای متن و تصویر — کاملاً محلی و آفلاین.

دو فضای برداری مستقل داریم:

* فضای متنی  (۳۸۴ بعدی) — ``paraphrase-multilingual-MiniLM-L12-v2``
  برای متن اسناد و متنِ OCR شده‌ی عکس‌ها؛ چندزبانه و مناسب فارسی.
* فضای تصویری (۵۱۲ بعدی) — ``clip-vit-base-patch32``
  برای محتوای دیداری عکس‌ها. متنِ پرس‌وجو هم می‌تواند با انکودر متنیِ CLIP
  به همین فضا نگاشت شود (جست‌وجوی متن↔تصویر).

مدل‌ها با بارگذاری تنبل (lazy) لود می‌شوند تا اجرای CLI سبک بماند.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Sequence

from .config import Settings, get_settings
from .models import ensure_model

logger = logging.getLogger(__name__)

TEXT_EMBEDDING_DIM = 384
IMAGE_EMBEDDING_DIM = 512

# محدودیت انکودر متنی CLIP
_CLIP_MAX_TOKENS = 77

Vector = list[float]


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class Embedder:
    """بارگذاری مدل‌ها و تولید بردار برای متن و عکس.

    این کلاس thread-safe است (قفل روی بارگذاری مدل‌ها) و می‌تواند به‌عنوان
    یک نمونه‌ی مشترک بین Gradio و CLI استفاده شود.
    """

    def __init__(self, settings: Settings | None = None, *, preload: bool = False) -> None:
        self._settings = settings or get_settings()
        self._device = _resolve_device(self._settings.device)
        self._lock = threading.Lock()

        self._text_model = None      # SentenceTransformer
        self._clip_model = None      # CLIPModel
        self._clip_processor = None  # CLIPProcessor
        self._torch = None           # ماژول torch (پس از بارگذاری CLIP)

        if preload:
            self.load()

    # ------------------------------------------------------------------ #
    # بارگذاری مدل‌ها
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """هر دو مدل را (در صورت نیاز با دانلود) بارگذاری می‌کند."""
        self._ensure_text_model()
        self._ensure_clip()

    def _ensure_text_model(self):
        if self._text_model is not None:
            return self._text_model
        with self._lock:
            if self._text_model is None:
                local = ensure_model(self._settings.text_model_id, self._settings)
                from sentence_transformers import SentenceTransformer

                logger.info("بارگذاری مدل متنی روی %s", self._device)
                self._text_model = SentenceTransformer(str(local.path), device=self._device)
        return self._text_model

    def _ensure_clip(self):
        if self._clip_model is not None:
            return self._clip_model, self._clip_processor
        with self._lock:
            if self._clip_model is None:
                local = ensure_model(self._settings.image_model_id, self._settings)
                import torch
                from transformers import CLIPModel, CLIPProcessor

                logger.info("بارگذاری مدل تصویری روی %s", self._device)
                model = CLIPModel.from_pretrained(str(local.path))
                model.eval()
                model.to(self._device)
                self._clip_model = model
                self._clip_processor = CLIPProcessor.from_pretrained(str(local.path))
                self._torch = torch
        return self._clip_model, self._clip_processor

    # ------------------------------------------------------------------ #
    # متن — فضای MiniLM
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> Vector:
        """بردار متنیِ نرمال‌شده (۳۸۴ بعدی) برای یک رشته."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str], *, show_progress: bool = False) -> list[Vector]:
        """نسخه‌ی دسته‌ای ``embed_text`` — برای ایندکس‌کردن سریع‌تر."""
        if not texts:
            return []
        model = self._ensure_text_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [vector.tolist() for vector in vectors]

    # ------------------------------------------------------------------ #
    # تصویر — فضای CLIP
    # ------------------------------------------------------------------ #
    def embed_image(self, image_path: str | Path) -> Vector:
        """بردار تصویریِ نرمال‌شده (۵۱۲ بعدی) برای یک عکس."""
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[Vector]:
        """نسخه‌ی دسته‌ای ``embed_image``."""
        if not image_paths:
            return []
        from PIL import Image

        model, processor = self._ensure_clip()
        torch = self._torch

        images = []
        try:
            for path in image_paths:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB").copy())

            inputs = processor(images=images, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().tolist()
        finally:
            for image in images:
                image.close()

    def embed_text_for_image_space(self, text: str) -> Vector:
        """نگاشت متن به فضای برداری CLIP برای جست‌وجوی متن↔تصویر.

        توجه: انکودر متنیِ ``clip-vit-base-patch32`` عمدتاً انگلیسی است؛
        برای پرس‌وجوی فارسی نتیجه ضعیف‌تر است و در ``search`` با وزن کمتری
        ترکیب می‌شود.
        """
        model, processor = self._ensure_clip()
        torch = self._torch

        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_CLIP_MAX_TOKENS,
        ).to(self._device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()


_DEFAULT_EMBEDDER: Embedder | None = None
_DEFAULT_LOCK = threading.Lock()


def get_embedder(settings: Settings | None = None) -> Embedder:
    """نمونه‌ی مشترکِ ``Embedder`` (تا مدل‌ها دوبار در حافظه لود نشوند)."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_EMBEDDER is None:
                _DEFAULT_EMBEDDER = Embedder(settings)
    return _DEFAULT_EMBEDDER

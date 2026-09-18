"""جست‌وجوی معنایی روی آرشیو.

سه منبعِ رتبه‌بندی داریم و نتایج آن‌ها با **Reciprocal Rank Fusion** ترکیب
می‌شوند (چون شباهت کسینوسی در فضای متنی و فضای CLIP مقیاس یکسانی ندارد و
جمع وزنی مستقیم سوگیری ایجاد می‌کند):

1. ``text``       — بردار MiniLM پرس‌وجو در برابر متن اسناد و OCR عکس‌ها
2. ``clip_text``  — نگاشت پرس‌وجو به فضای CLIP برای یافتن عکس‌های مشابهِ مفهومی
3. ``clip_image`` — عکسِ نمونه‌ی کاربر در برابر بردار تصویری عکس‌ها

مورد ۱ و ۲ با هم «جست‌وجوی ترکیبی متن + عکس» را می‌سازند؛ با دادن عکس
نمونه، مورد ۳ هم اضافه می‌شود.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import TYPE_IMAGE, Settings, get_settings
from .embedder import Embedder, get_embedder
from .store import ArchiveStore, get_store

logger = logging.getLogger(__name__)

# ثابت استاندارد RRF
_RRF_K = 60
# ضریب تعداد کاندیداهایی که از هر منبع گرفته می‌شود
_CANDIDATE_FACTOR = 5
_MIN_CANDIDATES = 20

_PERSIAN_RE = re.compile(r"[؀-ۿ]")

SOURCE_TEXT = "text"
SOURCE_CLIP_TEXT = "clip_text"
SOURCE_CLIP_IMAGE = "clip_image"


@dataclass(slots=True)
class SearchResult:
    """یک نتیجه‌ی جست‌وجو (به ازای هر فایل، بهترین تکه)."""

    path: str
    filename: str
    type: str
    score: float
    text_snippet: str
    matched_by: list[str] = field(default_factory=list)
    similarities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_persian(text: str) -> bool:
    return bool(_PERSIAN_RE.search(text))


def _clip_text_weight(query: str) -> float:
    """انکودر متنی CLIP انگلیسی است؛ برای فارسی وزن کمتری می‌گیرد."""
    return 0.30 if _is_persian(query) else 0.65


def _similarity(distance: float) -> float:
    """فاصله‌ی کسینوسی Chroma (``1 - cos``) → شباهت در بازه‌ی ۰ تا ۱."""
    return max(0.0, min(1.0, 1.0 - distance))


def _snippet(hit: dict[str, Any], limit: int = 320) -> str:
    text = (hit.get("document") or hit.get("metadata", {}).get("extracted_text") or "").strip()
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


class SearchEngine:
    """موتور جست‌وجو با پشتیبانی از پرس‌وجوی متنی، تصویری و ترکیبی."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: ArchiveStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or get_embedder(self._settings)
        self._store = store or get_store(self._settings)

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str | None = None,
        *,
        image_path: str | Path | None = None,
        n_results: int = 5,
        type_filter: str | None = None,
        include_visual: bool = True,
    ) -> list[SearchResult]:
        """جست‌وجوی ترکیبی.

        :param query: پرس‌وجوی متنی (فارسی یا انگلیسی)
        :param image_path: عکس نمونه برای جست‌وجوی «شبیه این عکس»
        :param n_results: تعداد نتایج نهایی
        :param type_filter: ``"image"`` یا ``"document"`` برای محدودکردن نوع
        :param include_visual: استفاده از انکودر متنی CLIP برای یافتن عکس‌ها
        """
        query = (query or "").strip()
        if not query and image_path is None:
            return []

        candidates = max(_MIN_CANDIDATES, n_results * _CANDIDATE_FACTOR)
        where = {"type": type_filter} if type_filter else None

        ranked_lists: list[tuple[str, float, list[dict[str, Any]]]] = []

        if query:
            text_hits = self._store.query_text(
                self._embedder.embed_text(query), candidates, where
            )
            ranked_lists.append((SOURCE_TEXT, 1.0, text_hits))

            if include_visual and self._wants_images(type_filter):
                try:
                    clip_hits = self._store.query_image(
                        self._embedder.embed_text_for_image_space(query), candidates, None
                    )
                    ranked_lists.append((SOURCE_CLIP_TEXT, _clip_text_weight(query), clip_hits))
                except Exception as exc:  # CLIP نباید جست‌وجوی متنی را خراب کند
                    logger.warning("جست‌وجوی تصویری با متن ناموفق بود: %s", exc)

        if image_path is not None and self._wants_images(type_filter):
            image_hits = self._store.query_image(
                self._embedder.embed_image(image_path), candidates, None
            )
            ranked_lists.append((SOURCE_CLIP_IMAGE, 1.0, image_hits))

        return self._fuse(ranked_lists, n_results)

    @staticmethod
    def _wants_images(type_filter: str | None) -> bool:
        return type_filter in (None, TYPE_IMAGE)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fuse(
        ranked_lists: Sequence[tuple[str, float, list[dict[str, Any]]]],
        n_results: int,
    ) -> list[SearchResult]:
        """ادغام رتبه‌ها با RRF و یکتاسازی بر اساس مسیر فایل."""
        fused: dict[str, dict[str, Any]] = {}

        for source, weight, hits in ranked_lists:
            seen_paths: set[str] = set()
            rank = 0
            for hit in hits:
                metadata = hit.get("metadata") or {}
                path = metadata.get("path")
                if not isinstance(path, str):
                    continue
                # هر فایل فقط یک‌بار از هر منبع امتیاز می‌گیرد (بهترین تکه)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                rank += 1

                similarity = _similarity(hit["distance"])
                entry = fused.setdefault(
                    path,
                    {
                        "path": path,
                        "filename": metadata.get("filename", Path(path).name),
                        "type": metadata.get("type", "document"),
                        "snippet": _snippet(hit),
                        "score": 0.0,
                        "matched_by": [],
                        "similarities": {},
                    },
                )
                entry["score"] += weight / (_RRF_K + rank)
                entry["matched_by"].append(source)
                entry["similarities"][source] = round(similarity, 4)
                if source == SOURCE_TEXT or not entry["snippet"]:
                    entry["snippet"] = _snippet(hit) or entry["snippet"]

        if not fused:
            return []

        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:n_results]
        best = ordered[0]["score"] or 1.0

        return [
            SearchResult(
                path=item["path"],
                filename=item["filename"],
                type=item["type"],
                score=round(item["score"] / best, 4),
                text_snippet=item["snippet"],
                matched_by=item["matched_by"],
                similarities=item["similarities"],
            )
            for item in ordered
        ]


_DEFAULT_ENGINE: SearchEngine | None = None


def get_search_engine(settings: Settings | None = None) -> SearchEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = SearchEngine(settings)
    return _DEFAULT_ENGINE


def search(query: str, n_results: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
    """API ساده‌ی درخواستی: لیستی از دیکشنری‌ها.

    هر آیتم شامل ``path``، ``score``، ``text_snippet`` و ``type`` است
    (به‌علاوه‌ی ``filename``، ``matched_by`` و ``similarities``).
    """
    results: Iterable[SearchResult] = get_search_engine().search(query, n_results=n_results, **kwargs)
    return [result.to_dict() for result in results]


__all__ = ["SearchEngine", "SearchResult", "get_search_engine", "search"]

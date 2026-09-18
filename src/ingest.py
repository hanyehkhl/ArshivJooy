"""اسکن پوشه و ساخت ایندکس برداری.

قوانین:

* فایل‌های پشتیبانی‌شده: ``.pdf .txt .docx .jpg .jpeg .png``
* اسناد → استخراج متن (pypdf / python-docx / plain text)
* عکس‌ها → OCR با pytesseract **و هم‌زمان** بردار تصویری CLIP
* اگر فایل قبلاً با همان محتوا ایندکس شده باشد (مقایسه‌ی SHA-256)
  دوباره ایندکس نمی‌شود؛ اگر محتوا تغییر کرده باشد ردیف‌های قبلی حذف و
  از نو ساخته می‌شوند.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from .config import (
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    TYPE_DOCUMENT,
    TYPE_IMAGE,
    Settings,
    get_settings,
)
from .embedder import Embedder, get_embedder
from .extractors import (
    ExtractionError,
    chunk_text,
    extract_document_text,
    iter_supported_files,
    ocr_image,
)
from .store import ArchiveStore, StoredRecord, get_store

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

_HASH_CHUNK = 1 << 20  # 1 MiB


@dataclass(slots=True)
class IngestStats:
    """نتیجه‌ی یک عملیات ایندکس."""

    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    text_chunks: int = 0
    image_vectors: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary_fa(self) -> str:
        lines = [
            f"فایل‌های بررسی‌شده: {self.scanned}",
            f"ایندکس‌شده: {self.indexed}",
            f"رد‌شده (تکراری): {self.skipped}",
            f"ناموفق: {self.failed}",
            f"تکه‌های متنی: {self.text_chunks}",
            f"بردارهای تصویری: {self.image_vectors}",
            f"زمان: {self.elapsed_seconds:.1f} ثانیه",
        ]
        if self.errors:
            lines.append("")
            lines.append("خطاها:")
            lines.extend(f"  • {error}" for error in self.errors[:20])
            if len(self.errors) > 20:
                lines.append(f"  … و {len(self.errors) - 20} خطای دیگر")
        return "\n".join(lines)


def file_hash(path: Path) -> str:
    """SHA-256 محتوای فایل (به‌صورت جریانی، بدون بارکردن کل فایل)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_key(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


class Ingestor:
    """موتور ایندکس‌کردن آرشیو."""

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
    def ingest_folder(
        self,
        folder: str | Path,
        *,
        force: bool = False,
        show_progress: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestStats:
        """همه‌ی فایل‌های پشتیبانی‌شده‌ی زیر ``folder`` را ایندکس می‌کند."""
        started = time.perf_counter()
        root = Path(folder).expanduser().resolve()
        files = iter_supported_files(root, SUPPORTED_EXTENSIONS)

        stats = IngestStats(scanned=len(files))
        if not files:
            stats.elapsed_seconds = time.perf_counter() - started
            return stats

        known_hashes = {} if force else self._store.indexed_hashes()

        iterator = tqdm(
            files,
            desc="ایندکس آرشیو",
            unit="فایل",
            disable=not show_progress,
        )

        for index, path in enumerate(iterator, start=1):
            if progress_callback is not None:
                progress_callback(index, len(files), path.name)
            try:
                self._ingest_file(path, known_hashes, stats, force=force)
            except Exception as exc:  # هیچ فایلی نباید کل عملیات را متوقف کند
                stats.failed += 1
                message = f"{path.name}: {exc}"
                stats.errors.append(message)
                logger.warning("ایندکس %s ناموفق بود: %s", path, exc)

        stats.elapsed_seconds = time.perf_counter() - started
        return stats

    # ------------------------------------------------------------------ #
    def _ingest_file(
        self,
        path: Path,
        known_hashes: dict[str, str],
        stats: IngestStats,
        *,
        force: bool,
    ) -> None:
        absolute = str(path)
        content_hash = file_hash(path)

        if not force and known_hashes.get(absolute) == content_hash:
            stats.skipped += 1
            return

        if absolute in known_hashes:
            # محتوا عوض شده — ردیف‌های قدیمی باید بروند
            self._store.delete_by_path(absolute)

        is_image = path.suffix.lower() in IMAGE_EXTENSIONS
        item_type = TYPE_IMAGE if is_image else TYPE_DOCUMENT

        text = ocr_image(path, self._settings) if is_image else extract_document_text(path)
        if not text.strip():
            # حداقل نام فایل باید قابل جست‌وجو باشد
            text = path.stem.replace("_", " ").replace("-", " ")

        base_metadata = self._base_metadata(path, item_type, content_hash)
        text_records = self._build_text_records(path, text, base_metadata)
        self._store.upsert_text(text_records)
        stats.text_chunks += len(text_records)

        if is_image:
            image_record = self._build_image_record(path, text, base_metadata)
            self._store.upsert_image([image_record])
            stats.image_vectors += 1

        known_hashes[absolute] = content_hash
        stats.indexed += 1

    # ------------------------------------------------------------------ #
    def _base_metadata(self, path: Path, item_type: str, content_hash: str) -> dict[str, str | int | float]:
        stat = path.stat()
        return {
            "path": str(path),
            "filename": path.name,
            "type": item_type,
            "extension": path.suffix.lower(),
            "content_hash": content_hash,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "indexed_at": time.time(),
        }

    def _build_text_records(
        self,
        path: Path,
        text: str,
        base_metadata: dict[str, str | int | float],
    ) -> list[StoredRecord]:
        chunks = chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
            max_chunks=self._settings.max_chunks_per_file,
        )
        if not chunks:
            return []

        embeddings = self._embedder.embed_texts(chunks)
        key = _path_key(path)
        limit = self._settings.metadata_text_limit

        return [
            StoredRecord(
                id=f"{key}::t{index}",
                embedding=embedding,
                document=chunk,
                metadata={
                    **base_metadata,
                    "extracted_text": chunk[:limit],
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "space": "text",
                },
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

    def _build_image_record(
        self,
        path: Path,
        text: str,
        base_metadata: dict[str, str | int | float],
    ) -> StoredRecord:
        embedding = self._embedder.embed_image(path)
        limit = self._settings.metadata_text_limit
        return StoredRecord(
            id=f"{_path_key(path)}::image",
            embedding=embedding,
            document=text[:limit],
            metadata={
                **base_metadata,
                "extracted_text": text[:limit],
                "chunk_index": 0,
                "chunk_count": 1,
                "space": "image",
            },
        )


def ingest_folder(
    folder: str | Path,
    *,
    force: bool = False,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
    settings: Settings | None = None,
) -> IngestStats:
    """میان‌بر تابعی برای ایندکس‌کردن یک پوشه."""
    return Ingestor(settings).ingest_folder(
        folder,
        force=force,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )


__all__ = ["IngestStats", "Ingestor", "ingest_folder", "file_hash", "ExtractionError"]

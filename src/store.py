"""لایه‌ی دسترسی به ChromaDB.

دو کالکشن نگه‌داری می‌شود چون ابعاد بردارها متفاوت است:

* ``archive``        → بردار متنی ۳۸۴ بعدی (اسناد + متن OCR عکس‌ها)
* ``archive_images`` → بردار تصویری ۵۱۲ بعدی CLIP (یک ردیف به ازای هر عکس)

هیچ ``embedding_function`` به Chroma داده نمی‌شود؛ بردارها را خودمان
می‌سازیم. این کار از دانلود مدل پیش‌فرض ONNX توسط Chroma هم جلوگیری می‌کند
و شرط «کاملاً آفلاین» را حفظ می‌نماید.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import IMAGE_COLLECTION, TEXT_COLLECTION, Settings, get_settings

logger = logging.getLogger(__name__)

Metadata = dict[str, str | int | float | bool]


@dataclass(slots=True)
class StoredRecord:
    """یک ردیف آماده برای درج در Chroma."""

    id: str
    embedding: list[float]
    document: str
    metadata: Metadata


class ArchiveStore:
    """بسته‌بندی نازک روی Chroma با API متناسب با نیاز پروژه."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = self._build_client()
        self.text_collection = self._collection(TEXT_COLLECTION)
        self.image_collection = self._collection(IMAGE_COLLECTION)

    def _build_client(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self._settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # ------------------------------------------------------------------ #
    # نوشتن
    # ------------------------------------------------------------------ #
    @staticmethod
    def _upsert(collection, records: Sequence[StoredRecord]) -> None:
        if not records:
            return
        collection.upsert(
            ids=[record.id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.document for record in records],
            metadatas=[record.metadata for record in records],
        )

    def upsert_text(self, records: Sequence[StoredRecord]) -> None:
        self._upsert(self.text_collection, records)

    def upsert_image(self, records: Sequence[StoredRecord]) -> None:
        self._upsert(self.image_collection, records)

    def delete_by_path(self, path: str) -> None:
        """همه‌ی ردیف‌های مربوط به یک فایل را از هر دو کالکشن حذف می‌کند."""
        where = {"path": path}
        for collection in (self.text_collection, self.image_collection):
            try:
                collection.delete(where=where)
            except Exception as exc:  # pragma: no cover - وابسته به نسخه Chroma
                logger.warning("حذف ردیف‌های %s ناموفق بود: %s", path, exc)

    def reset(self) -> None:
        """پاک‌کردن کامل ایندکس."""
        for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self.text_collection = self._collection(TEXT_COLLECTION)
        self.image_collection = self._collection(IMAGE_COLLECTION)

    # ------------------------------------------------------------------ #
    # خواندن
    # ------------------------------------------------------------------ #
    def indexed_hashes(self) -> dict[str, str]:
        """نگاشت ``path -> content_hash`` برای همه‌ی فایل‌های ایندکس‌شده.

        یک‌بار در ابتدای ایندکس خوانده می‌شود تا در حلقه‌ی اصلی هیچ
        درخواستی به دیتابیس زده نشود.
        """
        result: dict[str, str] = {}
        for collection in (self.text_collection, self.image_collection):
            offset = 0
            page = 5000
            while True:
                try:
                    batch = collection.get(include=["metadatas"], limit=page, offset=offset)
                except TypeError:  # نسخه‌های قدیمی‌تر بدون limit/offset
                    batch = collection.get(include=["metadatas"])
                metadatas: Iterable[dict[str, Any]] = batch.get("metadatas") or []
                count = 0
                for metadata in metadatas:
                    count += 1
                    path = metadata.get("path")
                    content_hash = metadata.get("content_hash")
                    if isinstance(path, str) and isinstance(content_hash, str):
                        result[path] = content_hash
                if count < page:
                    break
                offset += page
        return result

    def query_text(
        self,
        embedding: Sequence[float],
        n_results: int,
        where: Metadata | None = None,
    ) -> list[dict[str, Any]]:
        return self._query(self.text_collection, embedding, n_results, where)

    def query_image(
        self,
        embedding: Sequence[float],
        n_results: int,
        where: Metadata | None = None,
    ) -> list[dict[str, Any]]:
        return self._query(self.image_collection, embedding, n_results, where)

    @staticmethod
    def _query(
        collection,
        embedding: Sequence[float],
        n_results: int,
        where: Metadata | None,
    ) -> list[dict[str, Any]]:
        if collection.count() == 0:
            return []
        response = collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, n_results),
            where=where or None,
            include=["metadatas", "documents", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        return [
            {
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": float(distances[i]) if i < len(distances) else 1.0,
            }
            for i in range(len(ids))
        ]

    # ------------------------------------------------------------------ #
    # آمار
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, int]:
        hashes = self.indexed_hashes()
        return {
            "text_chunks": self.text_collection.count(),
            "image_vectors": self.image_collection.count(),
            "files": len(hashes),
        }


_DEFAULT_STORE: ArchiveStore | None = None


def get_store(settings: Settings | None = None) -> ArchiveStore:
    """نمونه‌ی مشترکِ ``ArchiveStore``."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = ArchiveStore(settings)
    return _DEFAULT_STORE

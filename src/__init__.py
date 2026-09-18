"""ArshivJooy — موتور جست‌وجوی معنایی محلی و آفلاین برای آرشیو شخصی.

نام‌های عمومی به‌صورت تنبل (PEP 562) بارگذاری می‌شوند تا صرفِ ``import src.config``
کتابخانه‌های سنگینی مثل torch و chromadb را به حافظه نیاورد.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

_EXPORTS: dict[str, str] = {
    "Settings": "config",
    "get_settings": "config",
    "Embedder": "embedder",
    "get_embedder": "embedder",
    "Ingestor": "ingest",
    "IngestStats": "ingest",
    "ingest_folder": "ingest",
    "SearchEngine": "search",
    "SearchResult": "search",
    "get_search_engine": "search",
    "search": "search",
    "ArchiveStore": "store",
    "get_store": "store",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)

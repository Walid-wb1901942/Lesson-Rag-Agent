"""Lazy-loaded BM25 index built from the full Qdrant corpus.

The index is built once on first access, persisted to disk, and reloaded on
subsequent server restarts.  Call `invalidate_bm25_index()` if the corpus
changes and you need the index rebuilt from Qdrant.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from threading import Lock

from rank_bm25 import BM25Okapi

from app.config import settings
from app.services.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

_lock: Lock = Lock()
_instance: BM25Index | None = None

_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_index.pkl"


class BM25Index:
    """In-memory BM25 index over all document chunks in Qdrant."""

    def __init__(self, chunk_ids: list[str], corpus: list[list[str]]) -> None:
        self._chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs for the top-k hits."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._chunk_ids)


def _build_index() -> BM25Index:
    """Scroll all chunks from Qdrant and build a BM25 index."""
    client = get_qdrant_client()
    chunk_ids: list[str] = []
    corpus: list[list[str]] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=["text"],
            with_vectors=False,
        )
        for point in results:
            text = (point.payload or {}).get("text", "")
            chunk_ids.append(str(point.id))
            corpus.append(text.lower().split())
        if next_offset is None:
            break
        offset = next_offset

    logger.info("BM25 index built: %d chunks", len(chunk_ids))
    return BM25Index(chunk_ids=chunk_ids, corpus=corpus)


def get_bm25_index() -> BM25Index:
    """Return the singleton BM25 index.

    On first access within a process, tries to load from the on-disk cache.
    If no cache exists, builds from Qdrant and saves to disk for future restarts.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                if _CACHE_PATH.exists():
                    try:
                        with open(_CACHE_PATH, "rb") as f:
                            _instance = pickle.load(f)
                        logger.info("BM25 index loaded from cache (%d chunks)", _instance.size)
                    except Exception as exc:
                        logger.warning("BM25 cache load failed (%s); rebuilding from Qdrant", exc)
                        _instance = _build_index()
                        _save_cache(_instance)
                else:
                    _instance = _build_index()
                    _save_cache(_instance)
    return _instance


def _save_cache(index: BM25Index) -> None:
    """Persist the BM25 index to disk."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(index, f)
        logger.info("BM25 index saved to cache (%s)", _CACHE_PATH)
    except Exception as exc:
        logger.warning("BM25 cache save failed: %s", exc)


def invalidate_bm25_index() -> None:
    """Discard the in-memory index and delete the on-disk cache.

    The index will be rebuilt from Qdrant on the next access.
    """
    global _instance
    with _lock:
        _instance = None
        if _CACHE_PATH.exists():
            try:
                _CACHE_PATH.unlink()
                logger.info("BM25 cache deleted")
            except Exception as exc:
                logger.warning("BM25 cache delete failed: %s", exc)

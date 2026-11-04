from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, settings
from app.shop.catalog import load_products, product_text

INDEX_DIR = DATA_DIR / "catalog" / "index"
META_PATH = INDEX_DIR / "meta.json"
VEC_PATH = INDEX_DIR / "embeddings.npy"


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.catalog_embed_model, device=settings.embed_device)


@lru_cache
def _load_index() -> tuple[list[str], np.ndarray]:
    if not META_PATH.exists() or not VEC_PATH.exists():
        build_index()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    vectors = np.load(VEC_PATH)
    return meta["ids"], vectors


def build_index(force: bool = False) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not force and META_PATH.exists() and VEC_PATH.exists():
        return INDEX_DIR

    products = load_products()
    ids = [p["id"] for p in products]
    texts = [product_text(p) for p in products]
    model = _model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=settings.embed_batch_size,
        show_progress_bar=False,
    )
    np.save(VEC_PATH, np.asarray(vectors, dtype=np.float32))
    META_PATH.write_text(
        json.dumps(
            {
                "ids": ids,
                "model": settings.catalog_embed_model,
                "count": len(ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _load_index.cache_clear()
    return INDEX_DIR


def embed_search(query: str, limit: int = 8) -> list[tuple[str, float]]:
    if not query.strip():
        return []
    ids, vectors = _load_index()
    model = _model()
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = vectors @ np.asarray(q, dtype=np.float32)
    top = np.argsort(-scores)[:limit]
    return [(ids[i], float(scores[i])) for i in top]

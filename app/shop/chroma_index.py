from __future__ import annotations

from functools import lru_cache

from app.config import DATA_DIR, settings
from app.shop.catalog import load_products, product_text

PERSIST_DIR = str(DATA_DIR / "catalog" / "chroma")
COLLECTION = "product_catalog"


@lru_cache
def _collection():
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.catalog_embed_model,
        device=settings.embed_device,
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def build_chroma_index(force: bool = False) -> int:
    col = _collection()
    products = load_products()
    if force and col.count() > 0:
        # recreate
        import chromadb

        client = chromadb.PersistentClient(path=PERSIST_DIR)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        _collection.cache_clear()
        col = _collection()

    if col.count() >= len(products) and not force:
        return col.count()

    ids = [p["id"] for p in products]
    docs = [product_text(p) for p in products]
    metas = [
        {
            "title": p["title"],
            "brand": p["brand"],
            "category": p["category"],
            "retailer": p["retailer"],
            "price": float(p["price"]),
            "shipping": float(p["shipping"]),
        }
        for p in products
    ]
    # upsert in batches
    batch = 32
    for i in range(0, len(ids), batch):
        col.upsert(
            ids=ids[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )
    return col.count()


def chroma_search(query: str, limit: int = 8) -> list[tuple[str, float]]:
    col = _collection()
    if col.count() == 0:
        build_chroma_index()
        col = _collection()
    res = col.query(query_texts=[query], n_results=min(limit, max(col.count(), 1)))
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out = []
    for pid, dist in zip(ids, dists):
        # cosine distance → similarity
        sim = 1.0 - float(dist) if dist is not None else 0.0
        out.append((pid, sim))
    return out

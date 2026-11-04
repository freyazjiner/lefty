from __future__ import annotations

import json
import re
from functools import lru_cache

from app.config import ROOT_DIR, settings

CATALOG_PATH = ROOT_DIR / "data" / "catalog" / "products.json"


@lru_cache
def load_products() -> list[dict]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def product_text(p: dict) -> str:
    return " ".join(
        [
            p.get("title", ""),
            p.get("brand", ""),
            p.get("category", ""),
            " ".join(p.get("tags") or []),
        ]
    )


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def _token_hits(query_tokens: set[str], hay_tokens: set[str]) -> int:
    """Exact overlap, plus model-code substrings (xm5 ⊂ 1000xm5)."""
    hits = 0
    for qt in query_tokens:
        if qt in hay_tokens:
            hits += 1
            continue
        if len(qt) < 3:
            continue
        if any(qt in ht for ht in hay_tokens) or any(
            ht in qt for ht in hay_tokens if len(ht) >= 3
        ):
            hits += 1
    return hits


def _with_totals(p: dict, match_score: float) -> dict:
    item = dict(p)
    item["total"] = round(float(p["price"]) + float(p["shipping"]), 2)
    item["match_score"] = round(float(match_score), 4)
    return item


def _brand_boost(query: str, product: dict) -> float:
    brand = (product.get("brand") or "").lower().strip()
    if not brand:
        return 0.0
    q = query.lower()
    if brand in q or brand.split()[0] in _tokens(query):
        return 2.0
    return 0.0


def _category_boost(query: str, product: dict) -> float:
    """Prefer products whose category words appear in the query."""
    q = query.lower()
    cat = (product.get("category") or "").lower().replace("_", " ")
    aliases = {
        "headphones": ("headphones", "headset", "over-ear", "over ear"),
        "earbuds": ("earbuds", "earphones", "in-ear", "buds"),
        "shoes": ("shoes", "sneakers", "trainers"),
        "keyboard": ("keyboard",),
        "mouse": ("mouse",),
        "webcam": ("webcam", "camera for calls"),
        "laptop stand": ("laptop stand", "stand"),
        "backpack": ("backpack", "bag"),
        "smartwatch": ("watch", "smartwatch", "fitness tracker"),
        "monitor": ("monitor", "display"),
        "speaker": ("speaker",),
        "charger": ("charger", "power bank", "magsafe"),
        "tablet": ("tablet", "ipad"),
    }
    score = 0.0
    if cat and cat in q:
        score += 2.0
    for alias in aliases.get(cat, ()):
        if alias in q:
            score += 1.5
            break
    return score


def _phrase_boost(query: str, product: dict) -> float:
    """Reward contiguous title phrases present in the query."""
    title = (product.get("title") or "").lower()
    q = query.lower()
    score = 0.0
    # whole distinctive chunks from title
    for phrase in re.findall(r"[a-z0-9][a-z0-9\s\-]{2,}", title):
        phrase = phrase.strip(" -")
        if len(phrase) < 4:
            continue
        if phrase in q:
            score += 3.0 if len(phrase) >= 8 else 1.5
    # model tokens with digits (xm5, k2, s2722qc)
    for tok in _tokens(title):
        if any(c.isdigit() for c in tok) and len(tok) >= 2 and tok in q:
            score += 2.5
        elif any(c.isdigit() for c in tok) and len(tok) >= 4:
            for qt in _tokens(query):
                if len(qt) >= 3 and (qt in tok or tok in qt):
                    score += 2.0
                    break
    return score


def keyword_search(query: str, limit: int = 8) -> list[dict]:
    q = _tokens(query)
    if not q:
        return []

    by_id = {p["id"]: p for p in load_products()}
    scored: list[tuple[float, dict]] = []
    for p in by_id.values():
        hay = _tokens(product_text(p))
        overlap = _token_hits(q, hay)
        if overlap == 0:
            continue
        title_hits = _token_hits(q, _tokens(p.get("title", "")))
        score = (
            overlap
            + 0.5 * title_hits
            + _brand_boost(query, p)
            + _category_boost(query, p)
            + _phrase_boost(query, p)
        )
        scored.append((score, _with_totals(p, score)))

    scored.sort(key=lambda x: (-x[0], x[1]["total"]))
    return [item for _, item in scored[:limit]]


def embedding_search(query: str, limit: int = 8) -> list[dict]:
    from app.shop.embed_index import embed_search

    by_id = {p["id"]: p for p in load_products()}
    hits = embed_search(query, limit=max(limit * 2, 12))
    out = []
    for pid, score in hits:
        p = by_id.get(pid)
        if not p:
            continue
        boosted = (
            float(score)
            + 0.08 * _brand_boost(query, p)
            + 0.04 * _category_boost(query, p)
            + 0.03 * _phrase_boost(query, p)
        )
        out.append(_with_totals(p, boosted))
    out.sort(key=lambda x: (-x["match_score"], x["total"]))
    return out[:limit]


def hybrid_search(
    query: str,
    limit: int = 8,
    kw_weight: float | None = None,
    emb_weight: float | None = None,
    brand_bonus: float | None = None,
) -> list[dict]:
    """RRF over keyword + embedding, then absolute keyword blend for separation."""
    from app.shop.ranking import rrf_fuse, value_rerank

    kw_w = settings.hybrid_kw_weight if kw_weight is None else kw_weight
    emb_w = settings.hybrid_emb_weight if emb_weight is None else emb_weight
    brand_b = settings.hybrid_brand_bonus if brand_bonus is None else brand_bonus

    by_id = {p["id"]: p for p in load_products()}
    kw_hits = keyword_search(query, limit=limit * 2)
    emb_hits = embedding_search(query, limit=limit * 2)
    kw_ids = [p["id"] for p in kw_hits]
    emb_ids = [p["id"] for p in emb_hits]
    fused = rrf_fuse([kw_ids, emb_ids], weights=[kw_w, emb_w])

    kw_raw = {p["id"]: float(p["match_score"]) for p in kw_hits}
    score_map = {pid: score for pid, score in fused}
    for pid in list(score_map):
        p = by_id.get(pid)
        if not p:
            continue
        # Keep RRF for fusion, add absolute keyword signal so top ranks aren't flat.
        score_map[pid] += 0.02 * kw_raw.get(pid, 0.0)
        score_map[pid] += brand_b * _brand_boost(query, p)
        score_map[pid] += 0.01 * _phrase_boost(query, p)

    out = []
    for pid, score in sorted(score_map.items(), key=lambda x: -x[1]):
        p = by_id.get(pid)
        if p:
            out.append(_with_totals(p, score))
    return value_rerank(out)[:limit]


def search_products(query: str, limit: int | None = None) -> list[dict]:
    limit = limit or settings.catalog_top_k
    mode = (settings.catalog_search_mode or "hybrid").lower()
    if mode == "keyword":
        return keyword_search(query, limit=limit)
    if mode == "embedding":
        return embedding_search(query, limit=limit)
    if mode == "chroma":
        from app.shop.chroma_index import chroma_search

        by_id = {p["id"]: p for p in load_products()}
        hits = chroma_search(query, limit=limit)
        out = [_with_totals(by_id[pid], score) for pid, score in hits if pid in by_id]
        return out
    return hybrid_search(query, limit=limit)


def compare_for_budget(
    products: list[dict],
    budget: float | None,
) -> dict:
    enriched = []
    for p in products:
        total = round(float(p["price"]) + float(p["shipping"]), 2)
        row = dict(p)
        row["total"] = total
        row["within_budget"] = budget is None or total <= budget
        enriched.append(row)

    ranked = sorted(
        enriched,
        key=lambda p: (-float(p.get("match_score") or 0), float(p["total"])),
    )
    within = [p for p in ranked if p["within_budget"]]
    best = ranked[0] if ranked else None

    # Prefer in-budget only if relevance is close to the best match.
    # Tight gate: RRF ranks are often near-tied, so a loose ratio lets
    # weak cheap distractors beat the clear intended item.
    if best and within:
        best_score = float(best.get("match_score") or 0)
        threshold = best_score * 0.85
        strong_within = [
            p for p in within if float(p.get("match_score") or 0) >= threshold
        ]
        pool = strong_within if strong_within else [best]
    else:
        pool = within or ranked

    pool_sorted = sorted(
        pool,
        key=lambda p: (-float(p.get("match_score") or 0), float(p["total"])),
    )
    pick = pool_sorted[0] if pool_sorted else None
    alts = [p for p in ranked if not pick or p["id"] != pick["id"]][:3]
    all_over = bool(enriched) and not within and budget is not None
    if pick and budget is not None and float(pick["total"]) > float(budget):
        all_over = True

    return {
        "budget": budget,
        "candidates": enriched,
        "within_budget_count": len(within),
        "pick": pick,
        "alternatives": alts,
        "all_over_budget": all_over,
    }

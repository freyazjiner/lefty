from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked id lists."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for w, ranked in zip(weights, ranked_lists):
        for rank, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: -x[1])


def value_rerank(products: list[dict], budget: float | None = None) -> list[dict]:
    """Light re-rank: relevance first, then total cost, then free shipping."""
    def key(p: dict):
        total = float(p.get("total") or (float(p["price"]) + float(p["shipping"])))
        match = float(p.get("match_score") or 0)
        ship = float(p.get("shipping") or 0)
        under = 0 if budget is None or total <= budget else 1
        return (-match, under, total, ship)

    return sorted(products, key=key)

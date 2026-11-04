"""Sweep hybrid RRF weights; pick best by pick_accuracy (then recall, budget)."""

from __future__ import annotations

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

from app.config import ROOT_DIR, settings
from app.shop.catalog import compare_for_budget, hybrid_search
from app.shop.embed_index import build_index

CASES_PATH = ROOT_DIR / "data" / "eval" / "shop_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def score_weights(
    cases: list[dict],
    top_k: int,
    kw_w: float,
    emb_w: float,
    brand_b: float,
) -> dict:
    search_fn = partial(
        hybrid_search,
        kw_weight=kw_w,
        emb_weight=emb_w,
        brand_bonus=brand_b,
    )
    recall_hits = pick_hits = budget_ok = budget_n = 0

    for case in cases:
        expected = set(case.get("expected_pick_ids") or [])
        must = set(case.get("must_include_any") or expected)
        allow_over = bool(case.get("allow_over_budget"))
        budget = case.get("budget")

        candidates = search_fn(case["query"], limit=top_k)
        cand_ids = {c["id"] for c in candidates}
        recall_hits += int(bool(must & cand_ids))

        comparison = compare_for_budget(candidates, budget)
        pick = comparison.get("pick")
        pick_id = (pick or {}).get("id")
        pick_hits += int(pick_id in expected if expected else pick_id is not None)

        if budget is not None and pick is not None:
            budget_n += 1
            within = float(pick["total"]) <= float(budget)
            if within or (allow_over and comparison.get("all_over_budget")):
                budget_ok += 1

    n = len(cases) or 1
    return {
        "kw_weight": kw_w,
        "emb_weight": emb_w,
        "brand_bonus": brand_b,
        "recall_at_k": round(recall_hits / n, 4),
        "pick_accuracy": round(pick_hits / n, 4),
        "budget_compliance": round(budget_ok / budget_n, 4) if budget_n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hybrid RRF weights")
    parser.add_argument("--top-k", type=int, default=settings.catalog_top_k)
    parser.add_argument(
        "--kw",
        default="0.3,0.5,0.7,0.9,1.0,1.1,1.3",
        help="Comma-separated keyword RRF weights",
    )
    parser.add_argument(
        "--emb",
        default="0.3,0.5,0.7,0.9,1.0,1.1,1.3",
        help="Comma-separated embedding RRF weights",
    )
    parser.add_argument(
        "--brand",
        default="0.0,0.05,0.1",
        help="Comma-separated brand bonus scales",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT_DIR / "data" / "eval" / "weight_sweep.json"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write best weights into .env",
    )
    args = parser.parse_args()

    build_index(force=False)
    cases = load_cases()
    kw_vals = [float(x) for x in args.kw.split(",") if x.strip()]
    emb_vals = [float(x) for x in args.emb.split(",") if x.strip()]
    brand_vals = [float(x) for x in args.brand.split(",") if x.strip()]

    results = []
    for kw_w, emb_w, brand_b in itertools.product(kw_vals, emb_vals, brand_vals):
        row = score_weights(cases, args.top_k, kw_w, emb_w, brand_b)
        results.append(row)
        print(
            f"kw={kw_w:.2f} emb={emb_w:.2f} brand={brand_b:.2f}  "
            f"pick={row['pick_accuracy']:.1%}  "
            f"recall={row['recall_at_k']:.1%}  "
            f"budget={row['budget_compliance']}"
        )

    results.sort(
        key=lambda r: (
            r["pick_accuracy"],
            r["recall_at_k"],
            r["budget_compliance"] or 0,
        ),
        reverse=True,
    )
    best = results[0]
    print(
        "\n[best] "
        f"kw={best['kw_weight']} emb={best['emb_weight']} "
        f"brand={best['brand_bonus']}  "
        f"pick={best['pick_accuracy']:.1%} recall={best['recall_at_k']:.1%}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"best": best, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] wrote {out}")

    if args.apply:
        env_path = ROOT_DIR / ".env"
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        kv = {
            "LEFTY_HYBRID_KW_WEIGHT": str(best["kw_weight"]),
            "LEFTY_HYBRID_EMB_WEIGHT": str(best["emb_weight"]),
            "LEFTY_HYBRID_BRAND_BONUS": str(best["brand_bonus"]),
        }
        keys_seen = set()
        new_lines = []
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                k = line.split("=", 1)[0].strip()
                if k in kv:
                    new_lines.append(f"{k}={kv[k]}")
                    keys_seen.add(k)
                    continue
            new_lines.append(line)
        for k, v in kv.items():
            if k not in keys_seen:
                new_lines.append(f"{k}={v}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"[applied] updated {env_path}")


if __name__ == "__main__":
    main()

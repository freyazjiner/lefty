"""Retrieval + budget-pick eval (no LLM). Compares keyword vs hybrid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import ROOT_DIR, settings
from app.shop.catalog import (
    compare_for_budget,
    embedding_search,
    hybrid_search,
    keyword_search,
)
from app.shop.embed_index import build_index

CASES_PATH = ROOT_DIR / "data" / "eval" / "shop_cases.json"
MODES = {
    "keyword": keyword_search,
    "embedding": embedding_search,
    "hybrid": hybrid_search,
}


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def eval_mode(mode: str, cases: list[dict], top_k: int) -> dict:
    search_fn = MODES[mode]
    recall_hits = 0
    pick_hits = 0
    budget_ok = 0
    budget_n = 0
    rows = []

    for case in cases:
        query = case["query"]
        budget = case.get("budget")
        expected = set(case.get("expected_pick_ids") or [])
        must = set(case.get("must_include_any") or expected)
        allow_over = bool(case.get("allow_over_budget"))

        candidates = search_fn(query, limit=top_k)
        cand_ids = {c["id"] for c in candidates}
        recall = bool(must & cand_ids)
        recall_hits += int(recall)

        comparison = compare_for_budget(candidates, budget)
        pick = comparison.get("pick")
        pick_id = (pick or {}).get("id")
        pick_ok = pick_id in expected if expected else pick_id is not None
        pick_hits += int(pick_ok)

        if budget is not None and pick is not None:
            budget_n += 1
            within = float(pick["total"]) <= float(budget)
            if within or (allow_over and comparison.get("all_over_budget")):
                budget_ok += 1
                budget_pass = True
            else:
                budget_pass = False
        else:
            budget_pass = None

        rows.append(
            {
                "id": case["id"],
                "recall": recall,
                "pick_id": pick_id,
                "pick_ok": pick_ok,
                "budget_ok": budget_pass,
                "total": (pick or {}).get("total"),
            }
        )

    n = len(cases) or 1
    return {
        "mode": mode,
        "n": len(cases),
        "recall_at_k": round(recall_hits / n, 4),
        "pick_accuracy": round(pick_hits / n, 4),
        "budget_compliance": round(budget_ok / budget_n, 4) if budget_n else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lefty shopping retrieval eval")
    parser.add_argument(
        "--modes",
        default="keyword,hybrid",
        help="Comma-separated: keyword,embedding,hybrid",
    )
    parser.add_argument("--top-k", type=int, default=settings.catalog_top_k)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT_DIR / "data" / "eval" / "last_report.json"),
    )
    args = parser.parse_args()

    if args.rebuild_index or "hybrid" in args.modes or "embedding" in args.modes:
        print("[info] building catalog embedding index…")
        t0 = time.perf_counter()
        build_index(force=args.rebuild_index)
        print(f"[info] index ready in {time.perf_counter() - t0:.1f}s")

    cases = load_cases()
    reports = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode not in MODES:
            raise SystemExit(f"unknown mode: {mode}")
        print(f"[eval] mode={mode}")
        report = eval_mode(mode, cases, top_k=args.top_k)
        reports.append(report)
        print(
            f"  recall@{args.top_k}={report['recall_at_k']:.1%}  "
            f"pick_acc={report['pick_accuracy']:.1%}  "
            f"budget={report['budget_compliance']}"
        )
        for row in report["rows"]:
            mark = "✓" if row["pick_ok"] and row["recall"] else "✗"
            print(
                f"  {mark} {row['id']}: pick={row['pick_id']} "
                f"total={row['total']} budget_ok={row['budget_ok']}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

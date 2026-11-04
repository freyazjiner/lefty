# Lefty

Shopping agent side project: type a product + budget, get a recommendation from a local catalog that compares price + shipping.

I built this to apply what I’ve learned about AI agents and to study how retrieval works in practice. It’s not a live store integration — the catalog is a JSON file (~95 products).

## What it does

1. Parses your message with a local LLM (Ollama)
2. Searches the catalog (keyword + MiniLM embeddings, fused with RRF)
3. Compares candidates under budget with a relevance gate
4. Explains the pick in plain English

Flow: `parse → search → compare → reply` (plus clarify / chitchat paths).

**Example:** `Sony XM5 headphones under 450` → picks the in-budget XM5 listing with the best total (price + shipping), and says why.

## Results (offline eval)

| | keyword | embedding | **hybrid** |
|---|---:|---:|---:|
| recall@8 | 100% | 100% | **100%** |
| pick accuracy | 97.8% | 95.7% | **100%** |
| budget compliance | 97.8% | 100% | **100%** |

- Catalog: **95** mock CAD products
- Eval: **46** hard cases (brand/model traps, shipping vs budget, over-budget-only matches)
- Weights tuned with `scripts/sweep_hybrid_weights.py` instead of hand-guessing

## Quick start

**Prereqs:** Python 3.12+, [Ollama](https://ollama.com) with a local model (e.g. `left-cpu:latest`).

```bash
cd lefty
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m scripts.build_catalog_index
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000

API key for `/docs` calls: `lefty-dev-key`

## Eval

```bash
python -m scripts.run_eval --modes keyword,embedding,hybrid
python -m scripts.sweep_hybrid_weights --apply
```

## Project layout

```
app/
  api/       # FastAPI chat endpoint
  graph/     # LangGraph nodes
  shop/      # catalog search, embeddings, ranking
  static/    # chat UI
data/
  catalog/   # products.json + embedding index
  eval/      # labeled test cases
scripts/     # build index, run eval, sweep weights
```

## Next up

- Live retailer price APIs (with caching) instead of the mock CAD catalog
- Stronger local LLM parsing so fewer clarify loops
- Larger catalog + harder eval cases beyond the current 46
- Optional image / visual search

## Stack

Python, FastAPI, LangGraph, Ollama, sentence-transformers (MiniLM), Chroma, SQLite (chat logs only)

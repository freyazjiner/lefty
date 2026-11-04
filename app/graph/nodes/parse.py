import json
import re

from app.config import settings
from app.models.llm import get_llm_client

PARSE_PROMPT = """You extract a shopping request from the user message.

## User message
{user_input}

## History
{history}

## Output
JSON only, no markdown:
{{
  "intent": "shop_search | budget_check | chitchat | clarify | unknown",
  "confidence": 0.0,
  "product_query": "short product description for search, empty if unclear",
  "budget": null or number in CAD,
  "constraints": {{
    "brand": "",
    "color": "",
    "notes": ""
  }},
  "needs_clarify": true/false
}}

Rules:
- If they name a product or category to buy/compare, intent=shop_search
- Pull budget from phrases like "under $200", "budget 150", "max 100 CAD"
- product_query should be searchable keywords (brand + product type)
- needs_clarify=true if product_query is empty or too vague
"""


def parse_request(state: dict) -> dict:
    client = get_llm_client()

    history_lines = []
    for msg in state["messages"][-6:]:
        role = "User" if msg["role"] == "user" else "Agent"
        history_lines.append(f"{role}: {msg['content']}")
    history = "\n".join(history_lines) if history_lines else "(none)"

    prompt = PARSE_PROMPT.format(
        user_input=state["current_input"],
        history=history,
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )

    raw = (response.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    intent = "unknown"
    confidence = 0.0
    product_query = ""
    budget = None
    constraints: dict = {}
    needs_clarify = True

    try:
        data = json.loads(raw)
        intent = data.get("intent") or "unknown"
        confidence = float(data.get("confidence") or 0.0)
        product_query = (data.get("product_query") or "").strip()
        budget = data.get("budget")
        if budget is not None:
            budget = float(budget)
        constraints = data.get("constraints") or {}
        needs_clarify = bool(data.get("needs_clarify", not product_query))
    except (json.JSONDecodeError, TypeError, ValueError):
        product_query = state["current_input"]
        confidence = 0.3
        intent = "shop_search"
        needs_clarify = False
        m = re.search(
            r"(?:under|below|max|budget(?:\s*(?:of|is)?)?|upto|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
            state["current_input"],
            re.I,
        )
        if m:
            budget = float(m.group(1))

    # Regex budget fallback if LLM missed it
    if budget is None:
        m = re.search(
            r"(?:under|below|max|budget(?:\s*(?:of|is)?)?|upto|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
            state["current_input"],
            re.I,
        )
        if m:
            budget = float(m.group(1))

    if intent not in {
        "shop_search",
        "budget_check",
        "chitchat",
        "clarify",
        "unknown",
    }:
        intent = "unknown"

    if not product_query:
        intent = "clarify"
    elif intent in ("clarify", "unknown") and product_query:
        intent = "shop_search"
    elif needs_clarify and not product_query:
        intent = "clarify"

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "product_query": product_query,
        "budget": budget,
        "constraints": constraints,
    }

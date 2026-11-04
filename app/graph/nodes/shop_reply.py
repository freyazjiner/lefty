import json

from app.config import settings
from app.models.llm import get_llm_client

REPLY_PROMPT = """You are Lefty, a shopping agent. Be clear and practical.

## User ask
{user_input}

## Parsed request
query: {product_query}
budget_cad: {budget}

## Comparison (price + shipping = total)
{comparison}

## Rules
1. Recommend the best fit under budget when possible (use total, not sticker price alone)
2. Mention retailer, price, shipping, and total for the pick and 1–2 alternatives
3. If everything is over budget, say so and suggest the cheapest total or ask to raise budget
4. If no candidates, say you couldn't find a match in the catalog and ask for a clearer product name
5. Keep under ~180 words. No markdown tables. Do not invent products not in the comparison JSON.
"""


def generate_shop_reply(state: dict) -> dict:
    client = get_llm_client()
    comparison = state.get("comparison") or {}

    prompt = REPLY_PROMPT.format(
        user_input=state["current_input"],
        product_query=state.get("product_query") or "",
        budget=state.get("budget"),
        comparison=json.dumps(comparison, indent=2),
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    reply = (response.choices[0].message.content or "").strip()
    return {"reply": reply, "status": "active"}

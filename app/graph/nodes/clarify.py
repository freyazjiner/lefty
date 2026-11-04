from app.config import settings
from app.models.llm import get_llm_client

CLARIFY_PROMPT = """The shopping request is unclear. Ask one short clarifying question.

## User said
{user_input}

Ask for the product (and budget if missing). Offer 1–2 example phrasings.
One question only, under 40 words.
"""


def clarify_shop(state: dict) -> dict:
    client = get_llm_client()
    prompt = CLARIFY_PROMPT.format(user_input=state["current_input"])
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )
    reply = (response.choices[0].message.content or "").strip().strip('"').strip("'")
    return {
        "clarification_count": state.get("clarification_count", 0) + 1,
        "reply": reply,
        "status": "active",
    }

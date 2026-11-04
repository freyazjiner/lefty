from typing import Literal, TypedDict

IntentType = Literal[
    "shop_search",
    "budget_check",
    "chitchat",
    "clarify",
    "unknown",
]

SessionStatus = Literal["active", "completed"]


class AgentState(TypedDict):
    session_id: str
    user_id: str

    messages: list[dict]
    current_input: str

    intent: IntentType
    intent_confidence: float

    # Parsed shopping request
    product_query: str
    budget: float | None
    constraints: dict

    clarification_count: int

    candidates: list[dict]
    comparison: dict

    status: SessionStatus
    reply: str
    error: str | None
    node_timings: dict

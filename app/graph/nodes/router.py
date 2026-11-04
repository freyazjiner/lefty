def route_after_parse(state: dict) -> str:
    intent = state.get("intent") or "unknown"
    if intent == "clarify" or intent == "unknown":
        return "clarify"
    if intent == "chitchat":
        return "chitchat"
    return "search"

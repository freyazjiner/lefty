def handle_chitchat(state: dict) -> dict:
    return {
        "reply": (
            "Hey — I’m Lefty. Tell me what you want to buy and roughly what "
            "budget you’re working with (CAD), and I’ll dig through the catalog."
        ),
        "status": "active",
    }

from app.shop.catalog import compare_for_budget


def compare_options(state: dict) -> dict:
    comparison = compare_for_budget(
        state.get("candidates") or [],
        state.get("budget"),
    )
    return {"comparison": comparison}

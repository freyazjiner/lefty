from app.shop.catalog import search_products


def search_catalog(state: dict) -> dict:
    query_bits = [state.get("product_query") or state.get("current_input") or ""]
    constraints = state.get("constraints") or {}
    for key in ("brand", "color", "notes"):
        val = (constraints.get(key) or "").strip()
        if val:
            query_bits.append(val)
    query = " ".join(query_bits).strip()
    candidates = search_products(query, limit=8)
    return {"candidates": candidates}

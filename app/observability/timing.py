import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from app.observability.metrics import NODE_LATENCY


def timed_node(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            start = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = time.perf_counter() - start
            NODE_LATENCY.labels(node=name).observe(elapsed)
            timings = dict(state.get("node_timings") or {})
            timings[name] = round(elapsed, 4)
            if isinstance(result, dict):
                merged = dict(result)
                merged["node_timings"] = {**timings, **(result.get("node_timings") or {})}
                return merged
            return result

        return wrapper

    return decorator

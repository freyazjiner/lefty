from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter(
    "lefty_chat_requests_total",
    "Chat requests",
    ["status", "intent"],
)
ERRORS = Counter("lefty_chat_errors_total", "Chat errors")
LATENCY = Histogram(
    "lefty_chat_latency_seconds",
    "End-to-end chat latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)
NODE_LATENCY = Histogram(
    "lefty_node_latency_seconds",
    "Per-node latency",
    ["node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

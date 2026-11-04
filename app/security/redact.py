import re

PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ID_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b")
API_KEY_RE = re.compile(
    r"\b(?:sk|pk|api|key|token)[-_]?[A-Za-z0-9]{16,}\b",
    re.IGNORECASE,
)
PATH_RE = re.compile(r"(?:/Users|/home|/var|/etc)/[^\s]+")


def redact_text(text: str) -> str:
    if not text:
        return text
    out = PHONE_RE.sub("[REDACTED_PHONE]", text)
    out = EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    out = ID_RE.sub("[REDACTED_ID]", out)
    out = API_KEY_RE.sub("[REDACTED_SECRET]", out)
    out = PATH_RE.sub("[REDACTED_PATH]", out)
    return out


def find_sensitive(text: str) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    if PHONE_RE.search(text):
        hits.append("phone")
    if EMAIL_RE.search(text):
        hits.append("email")
    if ID_RE.search(text):
        hits.append("id")
    if API_KEY_RE.search(text):
        hits.append("api_key")
    if PATH_RE.search(text):
        hits.append("path")
    return hits


def sanitize_reply(text: str) -> tuple[str, list[str]]:
    hits = find_sensitive(text)
    if not hits:
        return text, []
    return redact_text(text), hits

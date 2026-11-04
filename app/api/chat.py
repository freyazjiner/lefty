import json
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db.models import ConversationTurn
from app.db.session import get_db
from app.graph.builder import get_agent_graph
from app.graph.state import AgentState
from app.observability.logging import get_logger
from app.observability.metrics import ERRORS, LATENCY, REQUESTS
from app.security.redact import redact_text, sanitize_reply

router = APIRouter()
log = get_logger("lefty.chat")

_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    confidence: float
    status: str
    product_query: str | None = None
    budget: float | None = None
    pick: dict | None = None
    candidate_count: int = 0


def _save_turn(payload: dict) -> None:
    db = next(get_db())
    try:
        db.add(ConversationTurn(**payload))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_api_key: str = Header(...),
):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    start = time.perf_counter()
    graph = get_agent_graph()

    session = _sessions.get(
        req.session_id,
        {"messages": [], "clarification_count": 0, "status": "active"},
    )
    session["messages"].append({"role": "user", "content": req.message})

    initial_state: AgentState = {
        "session_id": req.session_id,
        "user_id": req.user_id,
        "messages": session["messages"],
        "current_input": req.message,
        "intent": "unknown",
        "intent_confidence": 0.0,
        "product_query": "",
        "budget": None,
        "constraints": {},
        "clarification_count": session.get("clarification_count", 0),
        "candidates": [],
        "comparison": {},
        "status": "active",
        "reply": "",
        "error": None,
        "node_timings": {},
    }

    try:
        result = graph.invoke(initial_state)
    except Exception:
        ERRORS.inc()
        LATENCY.observe(time.perf_counter() - start)
        log.exception(
            "chat_error",
            session_id=req.session_id,
            user_input=redact_text(req.message),
        )
        raise HTTPException(status_code=500, detail="Internal error")

    elapsed = time.perf_counter() - start
    LATENCY.observe(elapsed)

    reply, hits = sanitize_reply(
        result.get("reply")
        or "Hmm, something broke on my side. Try rephrasing what you want and your budget."
    )
    if hits:
        log.warning("reply_sanitized", session_id=req.session_id, hits=hits)

    intent = result.get("intent", "unknown")
    confidence = float(result.get("intent_confidence", 0.0))
    status = result.get("status", "active")
    comparison = result.get("comparison") or {}
    pick = comparison.get("pick")
    candidates = result.get("candidates") or []
    node_timings = result.get("node_timings") or {}
    product_query = result.get("product_query") or None
    budget = result.get("budget")

    REQUESTS.labels(status=status, intent=intent).inc()

    if elapsed > settings.slow_turn_seconds:
        log.warning(
            "slow_turn",
            session_id=req.session_id,
            latency_s=round(elapsed, 3),
            node_timings=node_timings,
        )

    log.info(
        "chat_turn",
        session_id=req.session_id,
        intent=intent,
        product_query=product_query,
        budget=budget,
        pick_id=(pick or {}).get("id"),
        latency_s=round(elapsed, 3),
    )

    _save_turn(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "user_input": redact_text(req.message),
            "reply": redact_text(reply),
            "intent": intent,
            "confidence": round(confidence, 4),
            "status": status,
            "product_query": product_query,
            "budget": budget,
            "pick_id": (pick or {}).get("id"),
            "candidate_count": len(candidates),
            "node_timings": json.dumps(node_timings),
            "latency_ms": int(elapsed * 1000),
        }
    )

    session["messages"].append({"role": "assistant", "content": reply})
    session["clarification_count"] = result.get("clarification_count", 0)
    session["status"] = status
    _sessions[req.session_id] = session

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        intent=intent,
        confidence=round(confidence, 4),
        status=status,
        product_query=product_query,
        budget=budget,
        pick=pick,
        candidate_count=len(candidates),
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, **session}

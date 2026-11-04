from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ConversationTurn(Base):
    """One chat turn, stored for debugging / later training data."""

    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_id = Column(String(100), index=True, nullable=False)
    user_input = Column(Text, nullable=False)
    reply = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(30), nullable=False)
    product_query = Column(String(300), nullable=True)
    budget = Column(Float, nullable=True)
    pick_id = Column(String(50), nullable=True)
    candidate_count = Column(Integer, default=0)
    node_timings = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

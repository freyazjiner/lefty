from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LEFTY_",
        extra="ignore",
    )

    # LLM (Ollama by default)
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "left-cpu:latest"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_timeout: int = 60

    # Catalog search
    catalog_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "cpu"
    embed_batch_size: int = 16
    catalog_search_mode: str = "hybrid"  # hybrid | embedding | keyword | chroma
    catalog_top_k: int = 8
    hybrid_kw_weight: float = 0.3
    hybrid_emb_weight: float = 0.3
    hybrid_brand_bonus: float = 0.05

    # API
    database_url: str = f"sqlite:///{DATA_DIR / 'lefty.db'}"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_key: str = "lefty-dev-key"
    slow_turn_seconds: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

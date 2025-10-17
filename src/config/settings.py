from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv()


@dataclass(slots=True)
class Settings:
    index_dir: Path
    raw_data_dir: Path
    retriever_top_k: int
    embedding_model: str
    llm_provider: Literal["echo", "ollama", "openai"]
    llm_max_context_chunks: int

    llm_openai_api_key: str | None
    llm_openai_model: str
    llm_openai_base_url: str | None
    llm_openai_temperature: float

    llm_ollama_model: str
    llm_ollama_base_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        def resolve_path(env_key: str, default: str) -> Path:
            value = os.getenv(env_key, default)
            path = Path(value)
            if not path.is_absolute():
                return PROJECT_ROOT / path
            return path

        provider = os.getenv("RAG_LLM_PROVIDER", "echo").lower()
        if provider not in {"echo", "ollama", "openai"}:
            provider = "echo"

        return cls(
            index_dir=resolve_path("RAG_INDEX_DIR", "data/index"),
            raw_data_dir=resolve_path("RAG_RAW_DATA_DIR", "data/raw"),
            retriever_top_k=int(os.getenv("RAG_RETRIEVER_TOP_K", "5")),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_provider=provider,  # type: ignore[arg-type]
            llm_max_context_chunks=int(os.getenv("RAG_LLM_MAX_CONTEXT_CHUNKS", "4")),
            llm_openai_api_key=os.getenv("RAG_LLM_OPENAI_API_KEY"),
            llm_openai_model=os.getenv("RAG_LLM_OPENAI_MODEL", "gpt-4o-mini"),
            llm_openai_base_url=os.getenv("RAG_LLM_OPENAI_BASE_URL"),
            llm_openai_temperature=float(os.getenv("RAG_LLM_OPENAI_TEMPERATURE", "0.2")),
            llm_ollama_model=os.getenv("RAG_LLM_OLLAMA_MODEL", "llama3:8b"),
            llm_ollama_base_url=os.getenv("RAG_LLM_OLLAMA_BASE_URL", "http://localhost:11434"),
        )

@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()

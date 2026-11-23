"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SOP_DATA_DIR", PROJECT_ROOT / "data" / "sops")))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("SOP_STORAGE_DIR", PROJECT_ROOT / "storage")))
    log_file: Path = field(default_factory=lambda: Path(os.getenv("SOP_LOG_FILE", PROJECT_ROOT / "logs" / "query_log.jsonl")))

    # Embeddings: auto | huggingface | local
    embed_backend: str = field(default_factory=lambda: os.getenv("SOP_EMBED_BACKEND", "auto"))
    hf_model: str = field(default_factory=lambda: os.getenv("SOP_HF_MODEL", "BAAI/bge-small-en-v1.5"))

    # Optional LLM for wording the summaries: none | openai | ollama
    llm_backend: str = field(default_factory=lambda: os.getenv("SOP_LLM_BACKEND", "none"))
    openai_model: str = field(default_factory=lambda: os.getenv("SOP_OPENAI_MODEL", "gpt-4o-mini"))
    ollama_model: str = field(default_factory=lambda: os.getenv("SOP_OLLAMA_MODEL", "llama3.1"))

    # Retrieval
    top_k: int = 5                # passages shown for a normal question
    candidate_k: int = 15         # passages fetched from the vector index before re-ranking
    persist_index: bool = True    # save the vector index to storage/ and reuse it


def load_config(**overrides) -> AppConfig:
    cfg = AppConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config option: {key}")
        setattr(cfg, key, value)
    return cfg

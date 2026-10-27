"""Typed application configuration; environment access lives only here.

Read a fresh snapshot at application/factory boundaries instead of caching a
mutable process-wide singleton. Existing COURSE_RAG_* names remain supported.
No .env file is loaded implicitly, and secret fields are excluded from repr.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    web_data: Path = Path("data/web")
    llm_base_url: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    llm_model: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    ollama_batch_size: int = 128
    ollama_chat_model: str | None = None
    ollama_chat_num_predict: int = 512
    ollama_chat_num_ctx: int | None = None
    ollama_chat_timeout: float = 120.0
    ollama_chat_think: bool = False
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_model: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = field(default=None, repr=False)
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        values = {}
        integers = {"ollama_batch_size", "ollama_chat_num_predict", "ollama_chat_num_ctx"}
        paths = {"data_dir", "index_dir", "web_data"}
        for setting in fields(cls):
            name = "COURSE_RAG_" + setting.name.upper()
            raw = env.get(name)
            if raw is None:
                continue
            try:
                if setting.name in paths:
                    value = Path(raw)
                elif setting.name in integers:
                    if not raw and setting.name == "ollama_chat_num_ctx":
                        value = None
                    else:
                        value = int(raw)
                        if value <= 0:
                            raise ValueError
                elif setting.name == "ollama_chat_timeout":
                    value = float(raw)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError
                elif setting.name == "ollama_chat_think":
                    if raw.lower() not in {"true", "false"}:
                        raise ValueError
                    value = raw.lower() == "true"
                else:
                    value = raw
            except ValueError:
                # Do not include environment values, which may contain secrets.
                raise ValueError(f"Invalid configuration: {name}") from None
            values[setting.name] = value
        return cls(**values)


def get_settings() -> Settings:
    return Settings.from_env()

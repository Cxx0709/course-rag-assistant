from pathlib import Path

import pytest

from course_rag_assistant.answer import (
    OllamaChatGenerator,
    OpenAICompatibleChatGenerator,
    generator_from_env,
)
from course_rag_assistant.cli import build_parser
from course_rag_assistant.config import Settings, get_settings
from course_rag_assistant.dense import OllamaEmbedder, embedder_from_env
from course_rag_assistant.rerank import reranker_from_env


@pytest.fixture
def clean_env(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith("COURSE_RAG_"):
            monkeypatch.delenv(name)
    return monkeypatch


def test_defaults_preserve_existing_factory_behavior():
    settings = Settings.from_env({})
    assert settings.web_data == Path("data/web")
    assert settings.index_dir == Path("data/index")
    assert settings.ollama_batch_size == 128
    assert settings.ollama_chat_num_predict == 512
    assert settings.ollama_chat_num_ctx is None
    assert settings.ollama_chat_think is False
    assert generator_from_env(settings) is None
    assert embedder_from_env(settings) is None
    assert reranker_from_env(settings) is None


def test_typed_environment_values_and_secret_repr():
    settings = Settings.from_env({
        "COURSE_RAG_WEB_DATA": "custom/library", "COURSE_RAG_OLLAMA_CHAT_NUM_CTX": "16384",
        "COURSE_RAG_OLLAMA_CHAT_TIMEOUT": "42.5", "COURSE_RAG_OLLAMA_CHAT_THINK": "TRUE",
        "COURSE_RAG_LLM_API_KEY": "secret-chat", "COURSE_RAG_EMBEDDING_API_KEY": "secret-embed",
        "COURSE_RAG_RERANK_API_KEY": "secret-rerank", "UNRELATED": "ignored"})
    assert settings.web_data == Path("custom/library")
    assert settings.ollama_chat_num_ctx == 16384
    assert settings.ollama_chat_timeout == 42.5
    assert settings.ollama_chat_think is True
    assert "secret-" not in repr(settings)
    assert Settings.from_env({"COURSE_RAG_OLLAMA_CHAT_NUM_CTX": ""}).ollama_chat_num_ctx is None


@pytest.mark.parametrize("suffix,value", [
    ("OLLAMA_CHAT_NUM_PREDICT", "bad-private-value"), ("OLLAMA_CHAT_NUM_PREDICT", "0"),
    ("OLLAMA_CHAT_NUM_CTX", "-1"), ("OLLAMA_BATCH_SIZE", "0"),
    ("OLLAMA_CHAT_TIMEOUT", "nan"), ("OLLAMA_CHAT_TIMEOUT", "inf"),
    ("OLLAMA_CHAT_TIMEOUT", "-2"), ("OLLAMA_CHAT_THINK", "invalid")])
def test_invalid_values_fail_with_variable_name_only(suffix, value):
    with pytest.raises(ValueError) as exc:
        Settings.from_env({"COURSE_RAG_" + suffix: value})
    assert str(exc.value) == "Invalid configuration: COURSE_RAG_" + suffix


def test_factories_keep_ollama_priority_and_share_settings_snapshot():
    config = Settings(ollama_chat_model="local-chat", ollama_model="local-embed",
                      llm_base_url="https://example.invalid", llm_api_key="key", llm_model="hosted",
                      ollama_base_url="http://localhost:9999/", ollama_chat_num_predict=256,
                      ollama_chat_num_ctx=8192, ollama_chat_timeout=45, ollama_batch_size=16)
    chat = generator_from_env(config)
    assert isinstance(chat, OllamaChatGenerator)
    assert (chat.model, chat.num_predict, chat.num_ctx, chat.timeout) == ("local-chat", 256, 8192, 45)
    dense = embedder_from_env(config)
    assert isinstance(dense, OllamaEmbedder)
    assert dense.model == "local-embed" and dense.batch_size == 16
    assert chat.base_url == dense.base_url == "http://localhost:9999"


def test_hosted_partial_configuration_keeps_previous_contract():
    assert generator_from_env(Settings(llm_model="partial")) is None
    assert reranker_from_env(Settings(rerank_base_url="https://example.invalid")) is None
    with pytest.raises(ValueError, match="must be set together"):
        embedder_from_env(Settings(embedding_model="partial"))
    chat = generator_from_env(Settings(llm_base_url="https://example.invalid", llm_api_key="key", llm_model="hosted"))
    assert isinstance(chat, OpenAICompatibleChatGenerator)


def test_settings_are_not_cached_and_cli_paths_follow_configuration(clean_env):
    clean_env.setenv("COURSE_RAG_DATA_DIR", "custom/data")
    clean_env.setenv("COURSE_RAG_INDEX_DIR", "custom/index")
    first = get_settings()
    args = build_parser().parse_args(["search", "query"])
    assert args.data == Path("custom/data/samples")
    assert args.index == Path("custom/index/bm25.json")
    clean_env.setenv("COURSE_RAG_DATA_DIR", "next/data")
    assert first.data_dir == Path("custom/data")
    assert get_settings().data_dir == Path("next/data")

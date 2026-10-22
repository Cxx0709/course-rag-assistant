"""Constrained answer generation.

Contract:
- Every factual claim must be followed by a citation [doc, p.N] pointing at a
  retrieved passage that actually contains it.
- If no retrieved passage supports the answer, say so explicitly
  (abstention) instead of inventing content.
- Explanations may link to prerequisite concepts from the concept graph;
  links the user hasn't studied yet are surfaced as "review first".
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .retrieval import Hit


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate an answer from a constrained prompt."""


class OpenAICompatibleChatGenerator:
    """Small synchronous client for OpenAI-compatible chat APIs."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个严格依据课程资料回答的助手。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat API response does not contain message content") from exc


class OllamaChatGenerator:
    """Synchronous chat generator for a local Ollama model."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个严格依据课程资料回答的助手。"},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            return response.json()["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response does not contain message content") from exc


def generator_from_env() -> OpenAICompatibleChatGenerator | None:
    """Create the configured chat generator, or ``None`` for offline mode."""
    ollama_model = os.getenv("COURSE_RAG_OLLAMA_CHAT_MODEL")
    if ollama_model:
        return OllamaChatGenerator(
            ollama_model,
            os.getenv("COURSE_RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            float(os.getenv("COURSE_RAG_OLLAMA_CHAT_TIMEOUT", "120")),
        )
    base_url = os.getenv("COURSE_RAG_LLM_BASE_URL")
    api_key = os.getenv("COURSE_RAG_LLM_API_KEY")
    model = os.getenv("COURSE_RAG_LLM_MODEL")
    if not all((base_url, api_key, model)):
        return None
    return OpenAICompatibleChatGenerator(base_url, api_key, model)


@dataclass
class CitedAnswer:
    text: str                     # markdown with [doc, p.N] citations
    hits: list[Hit]               # passages the answer was built from
    unsupported_claims: list[str]  # claims the verifier could not anchor


def answer(question: str, hits: list[Hit], generator: TextGenerator | None = None) -> CitedAnswer:
    """Generate an answer constrained to the retrieved passages.

    With no generator this is an extractive deterministic baseline.  A supplied
    generator may write a smoother response, but its output always passes the
    same local citation verifier.
    """
    usable = [hit for hit in hits if hit.score > 0 and hit.chunk.text.strip()]
    if not usable:
        return CitedAnswer("资料中没有足够证据回答这个问题。", [], [question])

    context = usable[:3]
    if generator is not None:
        draft = CitedAnswer(generator.generate(build_prompt(question, context)), context, [])
        return verify_citations(draft)

    lines: list[str] = []
    for hit in context[:1]:
        citation = format_citation(hit)
        for sentence in _sentences(hit.chunk.text):
            if sentence:
                lines.append(f"{sentence} {citation}")
    draft = CitedAnswer("\n".join(lines), context[:1], [])
    return verify_citations(draft)


def verify_citations(ans: CitedAnswer) -> CitedAnswer:
    """Drop/flag claims whose citation does not match the cited passage.

    The strict pass requires a citation that resolves to one of ``ans.hits``
    and shares meaningful terms with the claim.  Unsupported claims are
    removed from the returned text and recorded for evaluation.
    """
    supported: list[str] = []
    unsupported = list(ans.unsupported_claims)
    for claim in _sentences(ans.text):
        citations = _parse_citations(claim)
        body = _CITATION_RE.sub("", claim).strip()
        cited_chunks = [chunk for citation in citations for chunk in _resolve(citation, ans.hits)]
        supported_by_source = any(_overlaps(body, source, strict=True) for source in cited_chunks)
        # Short paraphrases often share only one meaningful Chinese token;
        # retain them when the citation resolves to a retrieved passage.
        if not supported_by_source:
            supported_by_source = any(_overlaps(body, source, strict=False) for source in cited_chunks)
        if not body or not cited_chunks or not supported_by_source:
            if body and body not in unsupported:
                unsupported.append(body)
            continue
        supported.append(claim)
    if not supported:
        text = "资料中没有足够证据回答这个问题。"
    else:
        text = "\n".join(supported)
    return CitedAnswer(text, ans.hits, unsupported)


_CITATION_RE = re.compile(r"\[([^,\]]+),\s*(p\.\d+|section=[^\]]+)\]")


def format_citation(hit: Hit) -> str:
    """Format provenance for a chunk, using a section anchor for Markdown."""
    chunk = hit.chunk
    if chunk.page:
        return f"[{chunk.doc}, p.{chunk.page}]"
    section = chunk.section or "document"
    return f"[{chunk.doc}, section={section}]"


def build_prompt(question: str, hits: Iterable[Hit]) -> str:
    """Build the constrained prompt used by a future LLM generator."""
    passages = []
    for index, hit in enumerate(hits, start=1):
        passages.append(f"[{index}] {format_citation(hit)}\n{hit.chunk.text}")
    return (
        "只依据下面的课程资料回答。每个事实性句子都必须在句末复制对应的来源标记；"
        "资料没有支持时回答‘资料中没有足够证据回答这个问题。’\n\n"
        f"问题：{question}\n\n资料：\n" + "\n\n".join(passages)
    )


def _sentences(text: str) -> list[str]:
    # Do not split on the period in citation markers such as ``p.12``.
    parts: list[str] = []
    for part in re.split(r"(?<=[。！？!?])\s*|\n+", text):
        part = part.strip()
        if not part:
            continue
        if part.startswith("[") and parts:
            parts[-1] += " " + part
        else:
            parts.append(part)
    return parts


def _parse_citations(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), match.group(2)) for match in _CITATION_RE.finditer(text)]


def _resolve(citation: tuple[str, str], hits: list[Hit]) -> list[str]:
    doc, location = citation
    if location.startswith("p."):
        page = int(location[2:])
        return [hit.chunk.text for hit in hits if hit.chunk.doc == doc and hit.chunk.page == page]
    section = location.removeprefix("section=")
    return [hit.chunk.text for hit in hits if hit.chunk.doc == doc and hit.chunk.section == section]


def _overlaps(claim: str, source: str, strict: bool = True) -> bool:
    claim_terms = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", claim.lower()))
    source_terms = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", source.lower()))
    overlap = claim_terms & source_terms
    if not overlap:
        return False
    if not strict:
        return True
    return len(overlap) >= max(2, min(4, int(len(claim_terms) * 0.3 + 0.5)))


if __name__ == "__main__":
    import argparse

    from .ingest import ingest_dir
    from .retrieval import HybridRetriever

    parser = argparse.ArgumentParser(description="Answer a question from course materials")
    parser.add_argument("question")
    parser.add_argument("--data", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    retriever = HybridRetriever(ingest_dir(args.data))
    result = answer(args.question, retriever.search(args.question, top_k=8), generator=generator_from_env())
    print(result.text)
    if result.unsupported_claims:
        print(f"\n未通过引用校验的断言：{len(result.unsupported_claims)}")

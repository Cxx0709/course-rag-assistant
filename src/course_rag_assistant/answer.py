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

from dataclasses import dataclass

from .retrieval import Hit


@dataclass
class CitedAnswer:
    text: str                     # markdown with [doc, p.N] citations
    hits: list[Hit]               # passages the answer was built from
    unsupported_claims: list[str]  # claims the verifier could not anchor


def answer(question: str, hits: list[Hit]) -> CitedAnswer:
    """Generate an answer constrained to the retrieved passages.

    TODO (week 3): prompt the LLM with passages + citation format, then run the
    claim-level verifier (each cited claim must overlap its passage text).
    """
    raise NotImplementedError


def verify_citations(ans: CitedAnswer) -> CitedAnswer:
    """Drop/flag claims whose citation does not match the cited passage.

    TODO (week 3): implement text-overlap + optional LLM judge.
    """
    raise NotImplementedError

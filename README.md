# Course RAG Assistant

A course-focused RAG system that answers from *your own course materials* — with
page-exact citations, prerequisite-aware explanations, and a quiz loop that maps
wrong answers back to knowledge gaps.

## Why this is not "yet another PDF chatbot"

1. **Verifiable answers.** Every factual claim is anchored to a retrieved
   passage; answers cite the exact source file, page number, and paragraph.
   Claims without a supporting passage are flagged, not hallucinated.
2. **Prerequisite linking.** Concepts are linked in a dependency graph
   (e.g. "Naive Bayes → conditional independence → Bayes' theorem"), so
   answers point back to what you should review first.
3. **Cross-source comparison.** The same concept from different textbooks or
   lecture decks is aligned side-by-side, with definitions and notation
   differences made explicit.
4. **Error-gap analysis.** Quiz mode generates questions from the corpus,
   grades answers, and maps mistakes to missing concepts — producing a review
   list instead of just a score.
5. **Evaluated, not demoed.** A labeled question set measures retrieval
   accuracy, citation correctness, and abstention quality, with an ablation
   study (dense-only vs hybrid vs +rerank).

## Architecture

```
course materials (PDF / PPTX / MD, page numbers preserved)
        │ ingest.py: extract → clean → page-preserving chunk
        ▼
  chunk store (id, doc, page, section, concept tags)
        │ retrieval.py: BM25 + dense (BGE-M3) → BGE reranker
        ▼
  answer.py: constrained generation — every claim cites [doc p.N]
        │ evaluate.py: recall@k / MRR / citation precision / abstention
        ▼
  quiz mode → error-gap analysis → review list
```

## Status

Project scaffold + implementation plan. See
[`docs/implementation-plan.md`](docs/implementation-plan.md) for the full
6–8 week build plan, data design, and acceptance criteria.

## Quick start (planned)

```bash
pip install -e .
python -m course_rag_assistant.ingest  data/samples/
python -m course_rag_assistant.answer  "朴素贝叶斯为什么需要条件独立假设？"
python -m course_rag_assistant.evaluate eval/labeled-questions.jsonl
```

## Tech stack

Python, FastAPI · PyMuPDF ingestion · BGE-M3 embeddings · BM25 + dense hybrid
retrieval · BGE reranker · pgvector or Qdrant · OpenAI-compatible LLM API
(pluggable, local option supported)

## License

MIT

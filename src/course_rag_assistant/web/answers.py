"""Question answering and citation response presentation for the Web API."""
from pathlib import Path

from fastapi import HTTPException

from ..answer import _CITATION_RE, _parse_citations, _sentences, answer
from ..rerank import ChatReranker
from ..retrieval import HybridRetriever
from .prerequisites import learning_plans
from .schemas import AskRequest


class ObservedGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.failed = False

    def generate(self, prompt):
        try:
            return self.generator.generate(prompt)
        except Exception:
            self.failed = True
            raise


def ask_materials(payload: AskRequest, library, generator_factory, model_lock):
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "请输入你想问的问题。")
    documents = library.selected(payload.document_ids)
    base = library.retriever(tuple(sorted(d.id for d in documents)))
    try:
        generator = generator_factory() if payload.use_model else None
    except (ValueError, TypeError) as exc:
        raise HTTPException(503, "模型配置无法使用，请关闭模型辅助后重试。") from exc
    if payload.use_model and generator is None:
        raise HTTPException(409, "尚未配置模型，请关闭模型辅助后重试。")
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "正在处理上一个问题，请稍后再试。")
    try:
        trace = ObservedGenerator(generator) if generator else None
        # Uploaded materials have their own lexical index. Never attach the
        # existing textbook vectors to an unrelated uploaded document set.
        retriever = HybridRetriever(base.chunks, reranker=ChatReranker(generator)) if generator else base
        result = answer(question, retriever.search(question, 8), trace)
    finally:
        model_lock.release()
    by_source = {d.source: d for d in documents}
    sources, source_ids, claims = [], {}, []
    for line in _sentences(result.text):
        ids = []
        for source_name, location in _parse_citations(line):
            document = by_source.get(source_name)
            hits = [h for h in result.hits if h.chunk.doc == source_name and
                    (f"p.{h.chunk.page}" == location if location.startswith("p.") else
                     f"section={h.chunk.section or 'document'}" == location)]
            if not document or not hits:
                continue
            key = (source_name, location)
            if key not in source_ids:
                sid = str(len(sources) + 1)
                source_ids[key] = sid
                page = hits[0].chunk.page
                sources.append({"id": sid, "document_id": document.id, "name": document.name,
                                "page": page, "section": hits[0].chunk.section,
                                "text": "\n\n".join(dict.fromkeys(h.chunk.text for h in hits)),
                                "url": f"/api/documents/{document.id}/file" + (f"#page={page}" if page else ""),
                                "type": Path(document.source).suffix[1:]})
            if source_ids[key] not in ids:
                ids.append(source_ids[key])
        claims.append({"text": _CITATION_RE.sub("", line).strip(), "source_ids": ids})
    mode = "source" if not trace else "fallback" if trace.failed else "model"
    return {"question": question, "claims": claims, "sources": sources, "mode": mode,
            "learning_plans": learning_plans(documents, question=question) if not result.abstention_reason else [],
            "abstained": result.abstention_reason is not None,
            "reason": result.abstention_reason, "document_names": [d.name for d in documents]}

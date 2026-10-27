"""Present source-declared learning paths within the selected document scope."""
from ..prerequisites import PrerequisiteGraph


def learning_plans(documents, concepts=None, *, question=None):
    graph = PrerequisiteGraph([c for d in documents for c in d.chunks])
    roots = graph.matching_concepts(question) if question is not None else concepts or []
    by_source = {d.source: d for d in documents}
    plans = [graph.plan(concept) for concept in dict.fromkeys(roots)]
    for plan in plans:
        for item in [*plan["steps"], *plan["relations"]]:
            for citation in item["sources"]:
                document = by_source[citation["doc"]]
                citation.update(document_name=document.name, document_id=document.id,
                                type=document.public()["type"],
                                source_url=f"/api/documents/{document.id}/file" + (
                                    f"#page={citation['page']}" if citation["page"] else ""))
    return plans

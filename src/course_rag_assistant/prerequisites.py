"""Conservative prerequisite graphs from explicit declarations in course notes.

Supported declarations: ``先修概念：A、B。`` under a concept heading,
and ``学习X前，建议先复习Y、Z。``. Co-occurrence never creates an edge.
Edges point from a concept to its prerequisites. Source excerpts are retained
separately for the relation and for the material used to review each concept.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .ingest import Chunk


def concept_name(value: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*[.、\s]*", "", value).strip().strip("《》「」“”")


def is_prerequisite_statement(text: str) -> bool:
    """Recognize supported reading-order declarations, not concept explanations."""
    return bool(re.search(r"先修概念[：:]|学习[^。！？\n]+?前[，,](?:建议先复习|需要先理解)", text))


def source(chunk: Chunk, text: str | None = None) -> dict:
    return {"doc": chunk.doc, "page": chunk.page, "section": chunk.section,
            "text": chunk.text if text is None else text}


class PrerequisiteGraph:
    def __init__(self, chunks: list[Chunk]):
        self.edges: dict[str, dict[str, list[dict]]] = defaultdict(dict)
        self.materials: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            heading = concept_name(chunk.section.rsplit(" / ", 1)[-1])
            names = [concept_name(c) for c in chunk.concepts] or [heading]
            declarations = []
            in_fence = False
            for line in chunk.text.splitlines():
                line = line.strip()
                if re.match(r"^(?:```|~~~)", line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                match = re.fullmatch(r"先修概念[：:]\s*([^。！？\n]+)[。]?", line)
                if match and heading:
                    declarations.append((heading, match[1], line))
                for match in re.finditer(
                    r"(?:^|(?<=[。！？\n]))\s*学习([^。！？\n，,]+?)前[，,]"
                    r"(?:建议先复习|需要先理解)([^。！？\n]+)[。]", line
                ):
                    declarations.append((concept_name(match[1]), match[2], match[0].strip()))
            for dependent, prerequisites, evidence in declarations:
                for name in re.split(r"[、,，]", prerequisites):
                    name = concept_name(name)
                    if not 2 <= len(name) <= 40 or not 2 <= len(dependent) <= 40:
                        continue
                    citations = self.edges[dependent].setdefault(name, [])
                    item = source(chunk, evidence)
                    if item not in citations:
                        citations.append(item)
            # Declaration-only chunks are not a lesson explaining the concept.
            remaining = chunk.text
            for _, _, evidence in declarations:
                remaining = remaining.replace(evidence, "")
            if not remaining.strip():
                continue
            for name in names:
                if 2 <= len(name) <= 40:
                    item = source(chunk)
                    if item not in self.materials[name]:
                        self.materials[name].append(item)

    def matching_concepts(self, question: str) -> list[str]:
        # Prefer full, longer concept names over contained names.
        matched = []
        for name in sorted(self.edges, key=lambda n: (-len(n), n)):
            pattern = re.escape(name)
            if name.isascii():
                pattern = r"(?<!\w)" + pattern + r"(?!\w)"
            if re.search(pattern, question, re.IGNORECASE) and not any(name in n for n in matched):
                matched.append(name)
        return matched[:3]

    def plan(self, concept: str) -> dict:
        concept = concept_name(concept)
        ordered, visited, active, edges = [], set(), set(), []
        cycle = False
        truncated = False

        def visit(name: str):
            nonlocal cycle, truncated
            if name in active:
                cycle = True
                return
            if name in visited:
                return
            if len(visited) >= 32:
                truncated = True
                return
            visited.add(name)
            active.add(name)
            for prerequisite, evidence in self.edges.get(name, {}).items():
                edges.append({"concept": name, "prerequisite": prerequisite,
                              "sources": evidence[:3]})
                visit(prerequisite)
            active.remove(name)
            if name != concept:
                ordered.append(name)

        visit(concept)
        status = "cycle" if cycle else "truncated" if truncated else "ready" if edges else "not_declared"
        return {"concept": concept, "status": status, "relations": edges,
                "steps": [{"concept": name, "sources": self.materials.get(name, [])[:3],
                           "required_by": list(dict.fromkeys(e["concept"] for e in edges
                                                            if e["prerequisite"] == name))}
                          for name in ordered] if status == "ready" else []}

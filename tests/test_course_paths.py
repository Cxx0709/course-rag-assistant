import json
from pathlib import Path

import pytest

from course_rag_assistant.ingest import Chunk
from scripts import evaluate_course_paths as paths


def prepare(tmp_path, monkeypatch):
    (tmp_path / "source.pdf").write_bytes(b"placeholder")
    note = tmp_path / "notes.md"
    note.write_text("# 目标概念\n\n先修概念：基础概念。\n\n## 基础概念\n\n基础概念在这里解释。\n\n"
                    "参考：[教材](source.pdf#page=2)。", encoding="utf-8")
    monkeypatch.setattr(paths, "extract_pdf", lambda path: [Chunk("教材正文。", path.name, 2)])
    spec = {"note": {"path": note.name, "sha256": paths.digest(note)},
            "sources": [{"path": "source.pdf", "sha256": paths.digest(tmp_path / "source.pdf")}],
            "plans": [{"concept": "目标概念", "order": ["基础概念"]}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    return note, manifest, spec


def test_draft_validation_is_not_review_or_installation(tmp_path, monkeypatch):
    _, manifest, _ = prepare(tmp_path, monkeypatch)
    result = paths.evaluate(manifest, root=tmp_path)
    assert result["regression_passed"]
    assert result["review_status"] == "pending_human_review"
    assert not result["default_library_modified"]


def test_wrong_page_and_wrong_order_fail(tmp_path, monkeypatch):
    note, manifest, spec = prepare(tmp_path, monkeypatch)
    note.write_text(note.read_text(encoding="utf-8").replace("page=2", "page=99"), encoding="utf-8")
    spec["note"]["sha256"] = paths.digest(note)
    spec["plans"][0]["order"] = ["未知概念"]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    result = paths.evaluate(manifest, root=tmp_path)
    assert not result["regression_passed"]
    assert result["checks"] == {"references": False, "plans": False}


def test_modified_draft_requires_new_fingerprint(tmp_path, monkeypatch):
    note, manifest, _ = prepare(tmp_path, monkeypatch)
    note.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        paths.evaluate(manifest, root=tmp_path)


def test_actual_draft_supports_offline_web_learning_plan(tmp_path):
    from fastapi.testclient import TestClient

    from course_rag_assistant.web import create_app

    note = Path(__file__).resolve().parents[1] / "data/course-notes/机器学习复习路径-待审核.md"
    with TestClient(create_app(tmp_path, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        uploaded = client.post("/api/documents", params={"filename": note.name}, content=note.read_bytes())
        assert uploaded.status_code == 201
        doc_id = uploaded.json()["id"]
        response = client.post("/api/ask", json={"document_ids": [doc_id], "question": "什么是正则化？"})
        assert response.status_code == 200
        result = response.json()
        assert not result["abstained"]
        plan = result["learning_plans"][0]
        assert plan["concept"] == "正则化" and plan["status"] == "ready"
        assert [s["concept"] for s in plan["steps"]] == ["训练集", "开发集", "测试集", "训练误差与开发误差", "过拟合"]
        assert all(s["sources"] for s in plan["steps"])
        assert all(e["document_name"].endswith("待审核.md")
                   for row in [*plan["steps"], *plan["relations"]] for e in row["sources"])

import pytest
from fastapi.testclient import TestClient

from course_rag_assistant.web import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        yield client


def upload(client, text, name="notes.md"):
    response = client.post("/api/documents", params={"filename": name}, content=text.encode())
    assert response.status_code == 201
    return response.json()["id"]


def compare(client, ids, concept="条件独立"):
    return client.post("/api/compare", json={"concept": concept, "document_ids": ids})


def test_same_named_sources_keep_separate_evidence_and_links(client):
    a = upload(client, "# 条件独立\n\n公式记为 P(A,B|C)=P(A|C)P(B|C)。")
    b = upload(client, "# 条件独立\n\n公式记为 P(X,Y|Z)=P(X|Z)P(Y|Z)。")
    outside = upload(client, "# 条件独立\n\n不应出现的外部资料内容。")
    response = compare(client, [a, b])
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert [d["id"] for d in data["documents"]] == [a, b]
    row = next(r for r in data["rows"] if r["key"] == "notation")
    assert row["status"] == "different_excerpt"
    for index, doc in enumerate([a, b]):
        evidence = row["cells"][index][0]
        assert evidence["document_id"] == doc
        assert evidence["id"].startswith(f"{index + 1}-")
        assert evidence["source_url"] == f"/api/documents/{doc}/file"
        assert client.get(evidence["source_url"]).status_code == 200
        assert "".join(p["text"] for p in evidence["parts"]) == evidence["text"]
    assert outside not in response.text and "不应出现" not in response.text
    reversed_data = compare(client, [b, a]).json()
    assert reversed_data["documents"][0]["id"] == b


def test_partial_empty_invalid_scope_and_security(client):
    a = upload(client, "# 条件独立\n\n条件独立是指在给定条件下满足独立关系。")
    b = upload(client, "# 梯度下降\n\n梯度下降沿损失函数减小的方向更新参数。")
    assert compare(client, [a, b]).json()["status"] == "partial"
    assert compare(client, [a, b], "未知概念").json()["status"] == "not_found"
    assert compare(client, [a, a]).status_code == 400
    assert compare(client, [a]).status_code == 422
    assert compare(client, [a, b, a]).status_code == 422
    assert compare(client, [a, "missing"]).status_code == 404
    assert compare(client, [a, b], "   ").status_code == 400
    assert compare(client, [a, b], " a ").status_code == 400
    client.headers["Origin"] = "https://other.example"
    assert compare(client, [a, b]).status_code == 403
    client.headers.pop("Origin")
    client.headers.pop("X-Workspace-Token")
    assert compare(client, [a, b]).status_code == 403


def test_pdf_page_provenance_and_original_context(client):
    import pymupdf

    text = "条件独立是指在给定条件下满足独立关系。"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page().insert_text((72, 120), text, fontname="china-s", fontsize=11)
        uploaded = client.post("/api/documents?filename=lesson.pdf", content=pdf.tobytes())
    assert uploaded.status_code == 201
    other = upload(client, f"# 条件独立\n\n{text}")
    data = compare(client, [uploaded.json()["id"], other]).json()
    evidence = next(r for r in data["rows"] if r["key"] == "definition")["cells"][0][0]
    assert evidence["page"] == 2 and evidence["source_url"].endswith("#page=2")
    assert text in evidence["context"]
    assert client.get(evidence["source_url"]).headers["content-type"] == "application/pdf"


def test_role_alignment_has_scoped_sources_and_original_term_labels(client):
    a = upload(client, "# 数据选择\n\n为了选择模型，我们使用验证集。", "统计讲义.md")
    b = upload(client, "# 数据\n\n开发集用于调整参数。", "实战笔记.md")
    outside = upload(client, "# 开发集\n\n开发集用于选择模型，不能借用这份未选资料。")
    for query in ("开发集", "验证集"):
        response = compare(client, [a, b], query)
        data = response.json()
        assert data["status"] == "ready"
        assert data["alignment"]["status"] == "role_correspondence"
        quotes = [e for r in data["rows"] for cell in r["cells"] for e in cell]
        assert {e["matched_term"] for e in quotes} == {"开发集", "验证集"}
        assert {e["document_id"] for e in quotes} == {a, b}
        assert outside not in response.text
    unknown = upload(client, "# 验证集\n\n验证集用于检查数据格式。")
    data = compare(client, [b, unknown], "开发集").json()
    assert data["status"] == "needs_review"
    assert data["alignment"]["status"] == "insufficient_evidence"
    assert not any("parts" in e for r in data["rows"] for cell in r["cells"] for e in cell)


def test_variance_choice_filters_sources_and_validates_requests(client):
    a = upload(client, "# 方差\n\n方差的平方根被称为标准差。", "统计.md")
    b = upload(client, "# 方差\n\n方差指的是算法在测试集上的表现低于训练集的程度。", "算法.md")
    pending = compare(client, [a, b], "方差").json()
    assert pending["status"] == "needs_disambiguation" and pending["rows"] == []
    assert all(d["status"] == "pending" for d in pending["documents"])
    for sense, expected in [("random_variable_variance", ["found", "not_found"]),
                            ("model_error_variance", ["not_found", "found"])]:
        data = client.post("/api/compare", json={"concept": "方差", "document_ids": [a, b], "sense": sense}).json()
        assert data["status"] == "partial"
        assert [d["status"] for d in data["documents"]] == expected
        assert all(e["sense"] == sense for r in data["rows"] for cell in r["cells"] for e in cell)
    assert compare(client, [a, b], "随机变量的方差").json()["alignment"]["sense"] == "random_variable_variance"
    for concept, sense in [("方差", "unknown"), ("条件独立", "model_error_variance"),
                           ("随机变量的方差", "model_error_variance")]:
        response = client.post("/api/compare", json={"concept": concept, "document_ids": [a, b], "sense": sense})
        assert response.status_code == 400
    assert compare(client, [a, "missing"], "方差").status_code == 404

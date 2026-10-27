from fastapi.testclient import TestClient

from course_rag_assistant.web import create_app


def test_demo_quiz_grade_review_and_restart(tmp_path):
    app = create_app(tmp_path, generator_factory=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        doc = client.post("/api/demo").json()
        scope = {"document_ids": [doc["id"]], "count": 3}
        response = client.post("/api/quiz", json=scope)
        assert response.status_code == 201, response.text
        quiz = response.json()
        assert len(quiz["questions"]) >= 2
        assert "correct_index" not in str(quiz) and "sentence" not in str(quiz)
        choices = {q["id"]: 0 for q in quiz["questions"]}
        result = client.post(f"/api/quiz/{quiz['quiz_id']}/grade", json={"choices": choices})
        assert result.status_code == 200
        for row in result.json()["results"]:
            assert row["source_url"].startswith(f"/api/documents/{doc['id']}/file")
            assert row["document_name"] == doc["name"]
            assert client.get(row["source_url"]).status_code == 200
        review = client.post("/api/review", json=scope).json()
        assert len(review["gaps"]) == result.json()["total"] - result.json()["score"]
        assert client.post(f"/api/quiz/{quiz['quiz_id']}/grade", json={"choices": {"1": True}}).status_code == 422
        assert client.post("/api/quiz/missing/grade", json={"choices": {"1": 0}}).status_code == 404
        assert client.post("/api/quiz", json={"document_ids": ["missing"]}).status_code == 404
        client.headers.pop("X-Workspace-Token")
        assert client.post("/api/quiz", json=scope).status_code == 403
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        assert client.post("/api/review", json=scope).json() == review

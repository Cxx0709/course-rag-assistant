from fastapi.testclient import TestClient

from course_rag_assistant.quiz import QuizStore
from course_rag_assistant.web import create_app

NOTES = """# 机器学习
## 过拟合
先修概念：训练集与测试集。

过拟合是指模型在训练集上表现良好，但在新数据上表现较差。
## 梯度下降
梯度下降沿着损失函数减小的方向更新模型参数。
"""
PRIOR = "# 训练集与测试集\n\n训练集用于拟合模型参数，测试集用于评估模型在新数据上的表现。"


def test_answer_quiz_review_scope_and_restart(tmp_path):
    app = create_app(tmp_path, generator_factory=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        def upload(text):
            response = client.post("/api/documents?filename=notes.md", content=text.encode())
            assert response.status_code == 201
            return response.json()["id"]
        current, prior = upload(NOTES), upload(PRIOR)
        def ask(ids, question="什么是过拟合？"):
            response = client.post("/api/ask", json={"question": question, "document_ids": ids})
            assert response.status_code == 200
            return response.json()
        answer = ask([current, prior])
        assert not answer["abstained"]
        step = answer["learning_plans"][0]["steps"][0]
        assert step["concept"] == "训练集与测试集"
        assert step["sources"][0]["document_id"] == prior
        assert step["sources"][0]["source_url"] == f"/api/documents/{prior}/file"
        assert ask([current])["learning_plans"][0]["steps"][0]["sources"] == []
        assert ask([prior])["learning_plans"] == []
        assert ask([current], "火星的卫星有多少？")["learning_plans"] == []

        scope = {"document_ids": [current, prior], "count": 2}
        public = client.post("/api/quiz", json=scope).json()
        stored = QuizStore(tmp_path / "quiz.sqlite3").load(public["quiz_id"])
        choices = {q["id"]: (q["correct_index"] + (q["concept"] == "过拟合")) % len(q["options"])
                   for q in stored["questions"]}
        graded = client.post(f"/api/quiz/{public['quiz_id']}/grade", json={"choices": choices}).json()
        assert [p["concept"] for p in graded["learning_plans"]] == ["过拟合"]
        review = client.post("/api/review", json=scope).json()
        assert review["learning_plans"] == graded["learning_plans"]
        assert client.post("/api/review", json={"document_ids": [prior]}).json()["learning_plans"] == []
    with TestClient(create_app(tmp_path, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        assert client.post("/api/review", json=scope).json() == review
        retry = client.post("/api/quiz", json=scope).json()
        right = {q["id"]: q["correct_index"] for q in stored["questions"]}
        assert client.post(f"/api/quiz/{retry['quiz_id']}/grade", json={"choices": right}).json()["learning_plans"] == []
        assert client.post("/api/review", json=scope).json()["learning_plans"] == []

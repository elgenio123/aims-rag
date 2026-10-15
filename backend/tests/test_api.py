import pytest
from fastapi.testclient import TestClient

import src.api.app as api_module


class FakeIndexer:
    def __init__(self, fail=False):
        self.fail = fail

    def count(self):
        if self.fail:
            raise RuntimeError("db down")
        return 5


class FakePipeline:
    error = None
    db_fail = False

    def __init__(self):
        self.indexer = FakeIndexer(fail=FakePipeline.db_fail)

    def answer(self, question, top_k=4):
        if FakePipeline.error:
            raise FakePipeline.error
        return {
            "answer": f"echo: {question}",
            "trace": [{"doc_id": "d1", "source_url": "https://example.org/d1", "score": 0.9}],
            "used_chunks": top_k,
        }


@pytest.fixture
def client(monkeypatch):
    FakePipeline.error = None
    FakePipeline.db_fail = False
    monkeypatch.setattr(api_module, "RagPipeline", FakePipeline)

    def make():
        return TestClient(api_module.app)

    return make


def test_health_reports_chunk_count(client):
    with client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 5}


def test_health_is_503_when_database_is_down(client):
    FakePipeline.db_fail = True
    with client() as c:
        assert c.get("/health").status_code == 503


def test_ask_returns_answer_and_trace(client):
    with client() as c:
        r = c.post("/ask", json={"question": "  Where is AIMS?  ", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: Where is AIMS?"
    assert body["used_chunks"] == 3
    assert body["trace"][0]["source_url"] == "https://example.org/d1"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 9},
        {},
    ],
)
def test_ask_rejects_invalid_input(client, payload):
    with client() as c:
        assert c.post("/ask", json=payload).status_code == 422


def test_ask_hides_internal_errors(client):
    FakePipeline.error = RuntimeError("Unauthorized: key sk-or-secret-123")
    with client() as c:
        r = c.post("/ask", json={"question": "hello"})
    assert r.status_code == 502
    assert "sk-or" not in r.text

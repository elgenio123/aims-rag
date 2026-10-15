import os

# Must be set before `config` is imported. The unreachable base URL guarantees no test can spend credits.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENROUTER_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "8")
os.environ.setdefault("LOG_FILE", "/tmp/aims-rag-tests.log")

import psycopg
import pytest

KEYWORDS = ["admission", "scholarship", "library", "campus", "research", "fees", "housing", "exam"]


def fake_embed(texts):
    return [[t.lower().count(k) + 0.01 for k in KEYWORDS] for t in texts]


def make_chunk(cid: str, text: str, doc_id: str = "doc") -> dict:
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "source_url": f"https://example.org/{doc_id}",
        "category": "Admissions",
        "text": text,
        "token_count": len(text.split()),
    }


@pytest.fixture
def indexer(monkeypatch):
    from src.embedder.indexer import Indexer

    try:
        idx = Indexer()
    except psycopg.OperationalError as e:
        if os.getenv("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"Postgres not available: {e}")
    with idx._connect() as conn:
        conn.execute("TRUNCATE chunks")

    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    monkeypatch.setattr(idx, "embed_texts", counting_embed)
    idx.embed_calls = calls
    return idx

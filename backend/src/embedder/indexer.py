"""Embeddings via OpenRouter, vectors stored in Postgres + pgvector."""
import hashlib
from typing import List, Dict
from loguru import logger
import psycopg
from pgvector.psycopg import register_vector
from config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_X_TITLE,
)

COLUMNS = ("chunk_id", "doc_id", "source_url", "category", "text", "token_count")


def _clean(text: str) -> str:
    return text.replace("\x00", "")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Indexer:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for embeddings")
        from openai import OpenAI
        headers = {}
        if OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_X_TITLE:
            headers["X-Title"] = OPENROUTER_X_TITLE
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers or None,
            max_retries=5,
        )
        self.dim = EMBEDDING_DIMENSIONS
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(DATABASE_URL)
        register_vector(conn)
        return conn

    def _ensure_schema(self):
        with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            row = conn.execute(
                "SELECT a.atttypmod FROM pg_attribute a "
                "WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'embedding'"
            ).fetchone()
            if row is not None and row[0] != self.dim:
                raise RuntimeError(
                    f"Table 'chunks' stores {row[0]}-dim vectors but EMBEDDING_DIMENSIONS={self.dim}. "
                    "Drop the table (DROP TABLE chunks;) and re-run the index command."
                )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    text PRIMARY KEY,
                    doc_id      text NOT NULL,
                    source_url  text,
                    category    text,
                    text        text NOT NULL,
                    token_count integer,
                    content_hash text,
                    embedding   vector({self.dim}) NOT NULL
                )
                """
            )
            conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash text")
            conn.execute(
                "UPDATE chunks SET content_hash = encode(sha256(convert_to(text, 'UTF8')), 'hex') "
                "WHERE content_hash IS NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=self.dim,
        )
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def sync(self, chunks: List[Dict], force: bool = False) -> Dict[str, int]:
        """Make the table match `chunks` (the full corpus), embedding only new or edited chunks."""
        if not chunks:
            raise RuntimeError("No chunks to index -- is data/documents empty or not mounted?")
        with self._connect() as conn:
            existing = dict(conn.execute("SELECT chunk_id, content_hash FROM chunks").fetchall())

        current: Dict[str, Dict] = {}
        for c in chunks:
            text = _clean(c["text"])
            current[c["chunk_id"]] = {**c, "text": text, "content_hash": _hash(text)}

        todo = [c for cid, c in current.items() if existing.get(cid) != c["content_hash"]]
        new = sum(1 for c in todo if c["chunk_id"] not in existing)
        removed = [cid for cid in existing if cid not in current]
        stats = {"new": new, "changed": len(todo) - new, "removed": len(removed),
                 "unchanged": len(current) - len(todo)}
        logger.info(
            f"Index sync: {stats['new']} new, {stats['changed']} changed, "
            f"{stats['removed']} removed, {stats['unchanged']} unchanged"
        )

        # Re-embedding a wiped index costs credits, so a half-missing corpus must not prune silently.
        if existing and len(removed) > len(existing) / 2 and not force:
            raise RuntimeError(
                f"Sync would delete {len(removed)} of {len(existing)} indexed chunks. "
                "If the documents were really removed, re-run with: python main.py index --force"
            )

        if removed:
            with self._connect() as conn:
                conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (removed,))

        done = 0
        for start in range(0, len(todo), EMBEDDING_BATCH_SIZE):
            batch = todo[start:start + EMBEDDING_BATCH_SIZE]
            embs = self.embed_texts([c["text"] for c in batch])
            rows = [
                (c["chunk_id"], c["doc_id"], c.get("source_url"), c.get("category"),
                 c["text"], c.get("token_count"), c["content_hash"], emb)
                for c, emb in zip(batch, embs)
            ]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (chunk_id, doc_id, source_url, category, text, token_count, "
                        "content_hash, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET doc_id = EXCLUDED.doc_id, "
                        "source_url = EXCLUDED.source_url, category = EXCLUDED.category, "
                        "text = EXCLUDED.text, token_count = EXCLUDED.token_count, "
                        "content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding",
                        rows,
                    )
            done += len(batch)
            logger.info(f"Embedded and stored {done}/{len(todo)} chunks")
        return stats

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        q_emb = self.embed_texts([query])[0]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(COLUMNS)}, 1 - (embedding <=> %s::vector) AS score "
                "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (q_emb, q_emb, top_k),
            ).fetchall()
        return [dict(zip(COLUMNS + ("score",), r)) for r in rows]

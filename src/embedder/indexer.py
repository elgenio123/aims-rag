"""Embedding and vector index management using FAISS or Chroma."""
from typing import List, Dict, Tuple
from loguru import logger
from config import EMBEDDING_MODEL, VECTOR_DB_TYPE, VECTOR_DB_PATH, EMBEDDING_BACKEND
from pathlib import Path
import os

# FAISS backend
class FaissIndex:
    def __init__(self, dim: int):
        import faiss  # type: ignore
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = []
        self.metadata: List[Dict] = []

    def add(self, embeddings, metas: List[Dict]):
        import numpy as np
        vecs = np.array(embeddings).astype('float32')
        # Normalize for cosine similarity
        faiss = self.faiss
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(metas)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss_write = os.path.join(path, 'faiss.index')
        metas_write = os.path.join(path, 'meta.json')
        self.faiss.write_index(self.index, faiss_write)
        import json
        with open(metas_write, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index to {faiss_write}")

    def load(self, path: str):
        faiss_read = os.path.join(path, 'faiss.index')
        metas_read = os.path.join(path, 'meta.json')
        self.index = self.faiss.read_index(faiss_read)
        import json
        with open(metas_read, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded FAISS index from {faiss_read}")

    def search(self, query_emb, top_k: int = 4) -> List[Tuple[int, float]]:
        import numpy as np
        q = np.array([query_emb]).astype('float32')
        self.faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        # I[0] indexes and D[0] scores
        return [(int(i), float(d)) for i, d in zip(I[0], D[0])]

# Chroma backend
class ChromaIndex:
    def __init__(self, dim: int):
        import chromadb
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="aims_cameroon",
            metadata={"hnsw:space": "cosine"}
        )
        self.dim = dim
        self._count = 0

    def add(self, embeddings, metas: List[Dict]):
        ids = [m['chunk_id'] for m in metas]
        texts = [m['text'] for m in metas]
        metadatas = [{k: v for k, v in m.items() if k != 'text'} for m in metas]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        self._count += len(ids)
        logger.info(f"Inserted {len(ids)} chunks into Chroma collection")

    def save(self, path: str):
        # Chroma persists automatically
        logger.info(f"Chroma collection persisted at {VECTOR_DB_PATH}")

    def load(self, path: str):
        # Already persistent
        logger.info(f"Chroma collection loaded from {VECTOR_DB_PATH}")

    def search(self, query_emb, top_k: int = 4):
        res = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        ids = res.get('ids', [[]])[0]
        dists = res.get('distances', [[]])[0]
        return [(ids[i], dists[i]) for i in range(len(ids))]

class Indexer:
    def __init__(self):
        self.backend_name = EMBEDDING_BACKEND.lower()
        self.model = None
        self.dim = None
        # Ensure fastembed uses a persistent cache path to avoid /tmp eviction
        try:
            cache_root = Path(VECTOR_DB_PATH).parent / "fastembed_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_root))
        except Exception:
            pass
        if self.backend_name == 'fastembed':
            try:
                from fastembed import TextEmbedding
            except Exception as e:
                raise RuntimeError("fastembed is not installed. Install it or set EMBEDDING_BACKEND=sentence-transformers.") from e
            self.fe = TextEmbedding(model_name=EMBEDDING_MODEL)
            emb = next(self.fe.embed(["dim_probe"]))
            self.dim = len(emb)
            logger.info(f"Loaded fastembed model {EMBEDDING_MODEL} (dim={self.dim})")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self.dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded sentence-transformers model {EMBEDDING_MODEL} (dim={self.dim})")
            except Exception as e:
                logger.warning(f"sentence-transformers unavailable ({e}); falling back to fastembed.")
                from fastembed import TextEmbedding
                self.fe = TextEmbedding(model_name=EMBEDDING_MODEL)
                emb = next(self.fe.embed(["dim_probe"]))
                self.dim = len(emb)
                self.backend_name = 'fastembed'
                logger.info(f"Loaded fastembed model {EMBEDDING_MODEL} (dim={self.dim})")
        if VECTOR_DB_TYPE == 'faiss':
            self.backend = FaissIndex(self.dim)
        elif VECTOR_DB_TYPE == 'chroma':
            self.backend = ChromaIndex(self.dim)
        else:
            raise ValueError(f"Unsupported VECTOR_DB_TYPE: {VECTOR_DB_TYPE}")

    def embed_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        texts = [c['text'] for c in chunks]
        if self.backend_name == 'fastembed':
            embeddings = list(self.fe.embed(texts))  # numpy arrays
            embeddings = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=False)
            embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        return embeddings

    def add(self, chunks: List[Dict]):
        embs = self.embed_chunks(chunks)
        self.backend.add(embs, chunks)

    def save(self):
        self.backend.save(VECTOR_DB_PATH)

    def load(self):
        self.backend.load(VECTOR_DB_PATH)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.backend_name == 'fastembed':
            q = next(self.fe.embed([query]))
            q_emb = q.tolist() if hasattr(q, 'tolist') else list(q)
        else:
            q = self.model.encode([query], normalize_embeddings=False)[0]
            q_emb = q.tolist() if hasattr(q, 'tolist') else list(q)
        results = self.backend.search(q_emb, top_k=top_k)
        out = []
        if VECTOR_DB_TYPE == 'faiss':
            for idx, score in results:
                meta = self.backend.metadata[idx]
                meta_copy = dict(meta)
                meta_copy['score'] = score
                out.append(meta_copy)
        else:
            # Chroma returns ids and distances; fetch metadatas
            res = self.backend.collection.get(ids=[r[0] for r in results])
            for i, rid in enumerate(res['ids']):
                meta = res['metadatas'][i]
                doc = res['documents'][i]
                meta_copy = dict(meta)
                meta_copy['text'] = doc
                meta_copy['score'] = results[i][1]
                out.append(meta_copy)
        return out

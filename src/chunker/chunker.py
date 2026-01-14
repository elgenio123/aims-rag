"""Semantic chunker preserving sentence boundaries and metadata."""
from typing import List, Dict
import re
import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Use OpenAI cl100k_base tokenizer for consistent token budgeting
# This is an approximation for sentence-transformers embeddings
_enc = tiktoken.get_encoding("cl100k_base")

def _sentence_split(text: str) -> List[str]:
    # Simple sentence splitter by punctuation, preserving abbreviations
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]

def _token_len(text: str) -> int:
    return len(_enc.encode(text))

def chunk_document(doc: Dict) -> List[Dict]:
    """Chunk a document dict into overlapping token chunks preserving sentences."""
    sentences = _sentence_split(doc["raw_text"]) if doc.get("raw_text") else []
    chunks: List[Dict] = []

    current: List[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sent in sentences:
        sent_tokens = _token_len(sent)
        if current_tokens + sent_tokens <= CHUNK_SIZE:
            current.append(sent)
            current_tokens += sent_tokens
        else:
            # finalize current
            if current:
                text = " ".join(current)
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_chunk_{chunk_idx}",
                    "doc_id": doc["doc_id"],
                    "source_url": doc["source_url"],
                    "category": doc["category"],
                    "text": text,
                    "token_count": _token_len(text),
                })
                chunk_idx += 1

            # start new chunk with overlap
            overlap_tokens = 0
            overlap_sents: List[str] = []
            for prev in reversed(current):
                if overlap_tokens + _token_len(prev) <= CHUNK_OVERLAP:
                    overlap_sents.insert(0, prev)
                    overlap_tokens += _token_len(prev)
                else:
                    break
            current = overlap_sents + [sent]
            current_tokens = sum(_token_len(s) for s in current)

    # finalize last
    if current:
        text = " ".join(current)
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{chunk_idx}",
            "doc_id": doc["doc_id"],
            "source_url": doc["source_url"],
            "category": doc["category"],
            "text": text,
            "token_count": _token_len(text),
        })

    return chunks

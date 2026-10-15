from config import CHUNK_SIZE
from src.chunker import chunk_document


def _doc(text: str) -> dict:
    return {"doc_id": "d1", "source_url": "https://example.org/d1", "category": "Academics", "raw_text": text}


def test_empty_document_has_no_chunks():
    assert chunk_document(_doc("")) == []


def test_chunks_are_sequential_budgeted_and_overlapping():
    text = " ".join(f"Sentence number {i} describes the AIMS master's programme in detail." for i in range(300))
    chunks = chunk_document(_doc(text))

    assert len(chunks) > 2
    assert [c["chunk_id"] for c in chunks] == [f"d1_chunk_{i}" for i in range(len(chunks))]
    for c in chunks:
        assert c["token_count"] <= CHUNK_SIZE
        assert c["doc_id"] == "d1"
        assert c["source_url"] == "https://example.org/d1"
        assert c["category"] == "Academics"

    first_sentence_of_second = chunks[1]["text"].split(". ")[0]
    assert first_sentence_of_second in chunks[0]["text"]

import pytest

from conftest import make_chunk

CORPUS = [
    make_chunk("a_chunk_0", "Admission requires a bachelor degree. Admission deadlines are in March.", "a"),
    make_chunk("b_chunk_0", "The scholarship covers tuition, housing and travel.", "b"),
    make_chunk("c_chunk_0", "The library opens at 8am and the campus is in Limbe.", "c"),
    make_chunk("d_chunk_0", "Research groups work on data science and epidemiology research.", "d"),
]


def _embedded_ids(indexer):
    return [t for batch in indexer.embed_calls for t in batch]


def test_first_sync_embeds_everything(indexer):
    stats = indexer.sync(CORPUS)
    assert stats == {"new": 4, "changed": 0, "removed": 0, "unchanged": 0}
    assert len(_embedded_ids(indexer)) == 4
    assert indexer.count() == 4


def test_resync_of_unchanged_corpus_makes_no_embedding_calls(indexer):
    indexer.sync(CORPUS)
    indexer.embed_calls.clear()

    stats = indexer.sync(CORPUS)

    assert stats == {"new": 0, "changed": 0, "removed": 0, "unchanged": 4}
    assert indexer.embed_calls == []


def test_only_added_and_edited_chunks_are_embedded(indexer):
    indexer.sync(CORPUS)
    indexer.embed_calls.clear()

    edited = make_chunk("b_chunk_0", "The scholarship now also covers exam fees.", "b")
    added = make_chunk("e_chunk_0", "Student housing is on campus.", "e")
    stats = indexer.sync([CORPUS[0], edited, CORPUS[2], CORPUS[3], added])

    assert stats == {"new": 1, "changed": 1, "removed": 0, "unchanged": 3}
    assert sorted(_embedded_ids(indexer)) == sorted([edited["text"], added["text"]])
    with indexer._connect() as conn:
        text = conn.execute("SELECT text FROM chunks WHERE chunk_id = 'b_chunk_0'").fetchone()[0]
    assert text == edited["text"]


def test_removed_chunks_are_deleted_without_embedding(indexer):
    indexer.sync(CORPUS)
    indexer.embed_calls.clear()

    stats = indexer.sync(CORPUS[:3])

    assert stats["removed"] == 1
    assert indexer.embed_calls == []
    assert indexer.count() == 3


def test_mass_deletion_is_refused_unless_forced(indexer):
    indexer.sync(CORPUS)

    with pytest.raises(RuntimeError, match="would delete 3 of 4"):
        indexer.sync(CORPUS[:1])
    assert indexer.count() == 4

    indexer.sync(CORPUS[:1], force=True)
    assert indexer.count() == 1


def test_empty_corpus_is_refused(indexer):
    indexer.sync(CORPUS)
    with pytest.raises(RuntimeError, match="No chunks"):
        indexer.sync([])
    assert indexer.count() == 4


def test_search_ranks_the_relevant_chunk_first(indexer):
    indexer.sync(CORPUS)

    results = indexer.search("When does the library open?", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c_chunk_0"
    assert set(results[0]) == {"chunk_id", "doc_id", "source_url", "category", "text", "token_count", "score"}
    assert 0 < results[0]["score"] <= 1.0001
    assert results[0]["score"] >= results[1]["score"]


def test_dimension_mismatch_is_reported(indexer):
    from src.embedder.indexer import Indexer

    with indexer._connect() as conn:
        conn.execute("DROP TABLE chunks")
        conn.execute("CREATE TABLE chunks (chunk_id text PRIMARY KEY, text text, embedding vector(3))")
    try:
        with pytest.raises(RuntimeError, match="3-dim vectors"):
            Indexer()
    finally:
        with indexer._connect() as conn:
            conn.execute("DROP TABLE chunks")
        Indexer()

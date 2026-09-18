# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependency management is `uv` (Python 3.12, lockfile `uv.lock`). `requirements.txt` exists but is unpinned and not the source of truth — `pyproject.toml` + `uv.lock` are.

```bash
uv sync                                    # create .venv from uv.lock
uv run main.py scrape --url <url> [<url>…] # crawl + store JSON documents
uv run main.py index                       # chunk all stored documents + build/save vector index
uv run main.py query "question"            # retrieve + answer, prints answer then per-chunk trace
uv run main.py full --url <url> --query "question"
```

All commands must run from the repo root: every module under `src/` does `from config import …`, which resolves to the root-level `config.py` as a top-level module.

There is no test suite, linter, or formatter configured in this repo.

## Architecture

Five stages, wired together only through files on disk — each stage reads what the previous one wrote, so they can be run and debugged independently:

1. `src/scraper/crawler.py` — BFS crawl (`MAX_DEPTH`, `time.sleep(SCRAPE_DELAY_SECONDS)` per URL, robots.txt honored via `RobotFileParser`). Stays within the netlocs of the start URLs. HTML and PDF both supported. Writes `Document` JSON to `data/documents/`.
2. `src/storage/` — `Document` is a Pydantic model; `DocumentStorage` is one JSON file per doc, named `{doc_id}.json`.
3. `src/chunker/chunker.py` — sentence-boundary chunking budgeted with tiktoken `cl100k_base` (`CHUNK_SIZE` / `CHUNK_OVERLAP` tokens).
4. `src/embedder/indexer.py` — `Indexer` combines an embedding backend with a vector store backend. Writes `data/vector_db/{faiss.index,meta.json}`.
5. `src/rag/pipeline.py` — `RagPipeline.answer()` retrieves top-k chunks, formats them with `doc_id`/`category`/`source_url` headers into the prompt, and returns `{answer, trace, used_chunks}`.

### Things that bite

- **`main.py` passes raw dicts, not `Document` objects, to the chunker.** `_load_all_documents()` re-reads the JSON files with `json.load` and hands dicts to `chunk_document`, which indexes them with `doc["raw_text"]` etc. Adding a field to `Document` does not make it available to chunks — chunk metadata is built explicitly in `chunk_document`.
- **`uv run main.py index` is additive, not idempotent.** `RagPipeline.__init__` loads the existing index, and `build_index` appends to it. Running `index` twice duplicates every chunk. Delete `data/vector_db/` to rebuild from scratch.
- **FAISS retrieval is positional.** `FaissIndex.metadata` is a plain list aligned by insertion order with the FAISS row ids; `Indexer.search` looks up `self.backend.metadata[idx]`. The index file and `meta.json` must always be written and read as a pair. Chroma's path is keyed by `chunk_id` instead and returns distances rather than similarity scores, so `score` means opposite things across the two backends.
- **`main.py` imports heavy modules lazily inside each `cmd_*`** so `scrape` doesn't pay for torch/faiss. Keep new heavy imports inside the command functions.
- **Console logging drops WARNING.** `setup_logging` filters `WARNING` out of the stderr sink; warnings only appear in `logs/pipeline.log`. If a run seems silent about a problem, check the log file.
- **`doc_id` is the URL slugified and truncated to 100 chars** (`re.sub(r'[^a-zA-Z0-9]+', '_', url)[:100]`) and doubles as the filename, so long URLs sharing a prefix collide and overwrite.
- `data/` and `logs/` are gitignored, so a fresh clone has no corpus or index.

### Configuration

All tunables live in `config.py`, read from `.env` via `python-dotenv`, with defaults in code (`.env.example` documents them).

- **LLM selection is implicit**: `OPENROUTER_API_KEY` wins if set (routed through `ChatOpenAI` against `OPENROUTER_BASE_URL`); otherwise `MISTRAL_API_KEY` with `ChatMistralAI`. Setting both silently uses OpenRouter. LLM construction is deferred to `_ensure_llm()` at query time, so credential errors surface only when answering.
- **`EMBEDDING_BACKEND`** (`sentence-transformers` | `fastembed`) selects how text is embedded; **`VECTOR_DB_TYPE`** (`faiss` | `chroma`) selects where vectors go. They are independent. sentence-transformers silently falls back to fastembed if it fails to load — the embedding dimension must stay consistent with the on-disk index, so changing `EMBEDDING_MODEL` or backend requires deleting `data/vector_db/`.
- **`VALID_CATEGORIES`** in `config.py` is the closed set. `categorize_content` (`src/utils/text_utils.py`) scores keyword counts per category and defaults to `Institutional`; the crawler also falls back to `Institutional` when `validate_category()` raises. Adding a category means editing both the list and the keyword map.

### Grounding policy

`SYSTEM_PROMPT` in `src/rag/pipeline.py` is sent both as a `SystemMessage` and inlined in the prompt template. The project's stated policy is that answers must come only from retrieved context and must say "The information is not available in the current documents." otherwise. Preserve this when touching prompts — every answer is expected to be traceable via the returned `trace` entries.

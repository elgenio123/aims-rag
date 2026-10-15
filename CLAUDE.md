# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

- `backend/` — all Python (scraper, chunker, embeddings, RAG pipeline, FastAPI app, scripts). Has its own `pyproject.toml` + `uv.lock` (Python 3.12, `uv`), `Dockerfile`, `.env`, and `data/`.
- `web/` — Next.js 15 App Router chat UI. `app/api/ask/route.ts` proxies to the backend (`API_URL`) and rate-limits per IP; the browser never talks to FastAPI directly.
- `docker-compose.yml` — `db` (pgvector/pgvector:pg17), `api`, `web` (published on `WEB_PORT`, default 80), and `indexer` (profile `tools`, one-shot CLI runs with `backend/data` mounted).
- `docker-compose.test.yml` — isolated project (`aims-rag-test`): throwaway tmpfs pgvector + the backend `test` image target. No `.env`, fake API key, unreachable base URL — tests can never spend credits.
- `infra/` — Terraform for the AWS 3-tier setup: VPC + 2 public subnets, security-group chain (ALB:80 → web:80 → api:8000 → db:5432, SSH from the auto-detected admin IP), 3 Ubuntu 24.04 EC2 instances (`aws_instance.server` keyed web/api/db, Docker via `user_data.sh`), and an ALB targeting web. Always run through `infra/tf.sh` (Terraform Docker image; reads `infra/aws.env` and `~/.ssh/aims-rag.pub`). State is local and gitignored. The AWS account is in an Organization whose SCP denies all regions except `eu-north-1` — don't change the region default. The same region holds an unrelated `postpilot` instance + Elastic IP; never touch resources not created by this Terraform.
- `deploy/` — `deploy.sh <key>` (local driver: stages `setup build test deploy smoke`; build/test run on the PC, the same images are `docker save | ssh docker load`-ed to the servers; web/api auto-rollback if deploy/smoke fails). Server addresses come from `tf.sh output` (or `INVENTORY=<file>`). `remote.sh` is the server half, streamed over ssh via `bash -s` — so it can't read stdin for data. `compose/{db,api,web}.yml` become `~/aims-rag/compose.yml` on each server with `.env` (per-tier secrets only) and `release.env`/`release.prev.env` (IMAGE_TAG for rollback). The root `docker-compose.yml` is for local dev only.

## Commands

The user does not want tooling installed on their host — everything runs through Docker, including lockfile updates.

```bash
docker compose up -d --build                                   # db + api + web
docker compose --profile tools run --rm indexer                # python main.py index
docker compose --profile tools run --rm indexer python main.py query "question"
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose --profile tools run --rm indexer python main.py scrape --url <url> --resume
docker run --rm -u "$(id -u):$(id -g)" -e UV_CACHE_DIR=/tmp/uvcache -v "$PWD/backend":/app -w /app ghcr.io/astral-sh/uv:python3.12-bookworm-slim uv lock
docker run --rm -u "$(id -u):$(id -g)" -e npm_config_cache=/tmp/npmcache -v "$PWD/web":/app -w /app node:22-alpine npm install --package-lock-only
```

Python code must run with `backend/` as the working directory: every module under `src/` does `from config import …`, which resolves to `backend/config.py` as a top-level module. Scripts under `backend/scripts/` insert `backend/` on `sys.path` themselves.

Tests live in `backend/tests/` (pytest; chunker, indexer sync against real pgvector with fake embeddings, API validation, grounding prompt). Run them with `docker compose -f docker-compose.test.yml run --rm --build tests` then `... down -v`. `tests/conftest.py` sets env defaults before `config` is imported and provides the `indexer` fixture (truncates `chunks`, swaps `embed_texts` for a keyword-count fake with 8 dims). The `backend/Dockerfile` has `base` → `test` / `runtime` targets; only `runtime` ships. No linter or formatter is configured.

## Architecture

Five stages; scraping/chunking communicate through files on disk, retrieval through Postgres:

1. `src/scraper/crawler.py` — BFS crawl (`MAX_DEPTH`, `time.sleep(SCRAPE_DELAY_SECONDS)` per URL, robots.txt honored via `RobotFileParser`). Stays within the netlocs of the start URLs. HTML and PDF both supported. Writes `Document` JSON to `data/documents/`.
2. `src/storage/` — `Document` is a Pydantic model; `DocumentStorage` is one JSON file per doc, named `{doc_id}.json`.
3. `src/chunker/chunker.py` — sentence-boundary chunking budgeted with tiktoken `cl100k_base` (`CHUNK_SIZE` / `CHUNK_OVERLAP` tokens).
4. `src/embedder/indexer.py` — `Indexer` embeds via OpenRouter's OpenAI-compatible `/embeddings` (`openai` SDK, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`) and stores rows in the Postgres `chunks` table (pgvector, HNSW cosine index). Creates the extension/table/index on construction.
5. `src/rag/pipeline.py` — `RagPipeline.answer()` retrieves top-k chunks, formats them with `doc_id`/`category`/`source_url` headers into the prompt, and returns `{answer, trace, used_chunks}`. `src/api/app.py` wraps it as `POST /ask` + `GET /health`.

### Things that bite

- **`main.py` passes raw dicts, not `Document` objects, to the chunker.** `_load_all_documents()` re-reads the JSON files with `json.load` and hands dicts to `chunk_document`, which indexes them with `doc["raw_text"]` etc. Adding a field to `Document` does not make it available to chunks — chunk metadata is built explicitly in `chunk_document`.
- **`index` is a full-corpus sync keyed on `chunk_id` + `content_hash` (sha256 of the stored text).** Only new or text-changed chunks are embedded; chunk_ids missing from the corpus are deleted. It refuses to delete more than half the table without `--force`, and refuses an empty corpus. Since chunk ids are positional (`{doc_id}_chunk_{n}`), editing early text in a doc re-embeds that doc's later chunks too — but never other docs. `deploy/remote_setup.sh` runs this on every deploy.
- **Embedding dimension is baked into the table.** Changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` requires `DROP TABLE chunks;` and a full re-index (costs credits); `Indexer` raises if they disagree. pgvector HNSW caps at 2000 dims.
- **Every query and every eval row calls the embeddings API** — `scripts/evaluate_rag.py` spends credits on both answers and embedding-similarity scoring.
- **`main.py` imports heavy modules lazily inside each `cmd_*`.** Keep new heavy imports inside the command functions.
- **Console logging drops WARNING.** `setup_logging` filters `WARNING` out of the stderr sink; warnings only appear in `logs/pipeline.log` (inside the container unless mounted).
- **`doc_id` is the URL slugified and truncated to 100 chars** (`re.sub(r'[^a-zA-Z0-9]+', '_', url)[:100]`) and doubles as the filename, so long URLs sharing a prefix collide and overwrite.
- `backend/data/` and `logs/` are gitignored, so a fresh clone has no corpus.
- The `indexer` service bind-mounts `backend/data`; pass `HOST_UID`/`HOST_GID` for commands that write there (scrape, eval), otherwise it runs as uid 1000.

### Configuration

All tunables live in `backend/config.py`, read from `backend/.env` via `python-dotenv` (docker compose loads the same file with `env_file`), with defaults in code (`backend/.env.example` documents them).

- **LLM selection is implicit**: `OPENROUTER_API_KEY` wins if set (routed through `ChatOpenAI` against `OPENROUTER_BASE_URL`); otherwise `MISTRAL_API_KEY` with `ChatMistralAI`. LLM construction is deferred to `_ensure_llm()` at query time. Embeddings always require `OPENROUTER_API_KEY`.
- **Database URL**: `DATABASE_URL` if set, else built from `POSTGRES_PASSWORD` + `POSTGRES_HOST` (compose sets `POSTGRES_HOST=db`).
- **`VALID_CATEGORIES`** in `config.py` is the closed set. `categorize_content` (`src/utils/text_utils.py`) scores keyword counts per category and defaults to `Institutional`; the crawler also falls back to `Institutional` when `validate_category()` raises. Adding a category means editing both the list and the keyword map.

### Grounding policy

`SYSTEM_PROMPT` in `src/rag/pipeline.py` is sent both as a `SystemMessage` and inlined in the prompt template. The project's stated policy is that answers must come only from retrieved context and must say "The information is not available in the current documents." otherwise. Preserve this when touching prompts — every answer is expected to be traceable via the returned `trace` entries. The web UI keys off that exact phrase to style "not in the documents" answers.

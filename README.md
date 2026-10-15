# AIMS Cameroon RAG

A question-answering assistant for AIMS Cameroon. It scrapes the institute's websites, indexes the content in Postgres/pgvector with OpenRouter embeddings, and answers questions with an LLM that only uses retrieved, cited sources. It's served by a FastAPI backend and a Next.js web app, all in Docker.

## Structure

```
aims-rag/
├── backend/                 # Python: scraper, chunker, embeddings, RAG, FastAPI
│   ├── src/
│   │   ├── api/app.py       # FastAPI: GET /health, POST /ask
│   │   ├── scraper/         # BFS crawler (HTML + PDF, robots.txt aware)
│   │   ├── storage/         # Document model + JSON storage
│   │   ├── chunker/         # sentence-boundary, token-budgeted chunks
│   │   ├── embedder/        # OpenRouter embeddings + pgvector store
│   │   └── rag/pipeline.py  # retrieval + grounded answer generation
│   ├── scripts/             # evaluation, QA generation, fine-tune data prep
│   ├── data/documents/      # scraped corpus (gitignored)
│   ├── main.py              # CLI: scrape | index | query | full
│   ├── config.py
│   ├── .env                 # your secrets (gitignored), from .env.example
│   └── Dockerfile
├── web/                     # Next.js chat UI (proxies /api/ask → backend)
│   └── Dockerfile
├── infra/                   # Terraform: VPC, security groups, 3 EC2 servers, ALB
├── deploy/                  # release pipeline + per-server compose files + AWS guide
└── docker-compose.yml
```

## Run locally (Docker only)

```bash
cp backend/.env.example backend/.env   # set OPENROUTER_API_KEY and POSTGRES_PASSWORD
docker compose up -d --build           # db + api + web, UI on http://localhost
docker compose --profile tools run --rm indexer   # embed backend/data/documents into Postgres
```

Port 80 is taken? Use `WEB_PORT=3000 docker compose up -d`.

The index command is an incremental sync. Each chunk's content hash is compared with Postgres: only new or edited chunks are embedded, removed ones are deleted, and an unchanged knowledge base costs zero API calls.

Other CLI commands run through the same image:

```bash
docker compose --profile tools run --rm indexer python main.py query "What are the admission requirements?"
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose --profile tools run --rm indexer \
  python main.py scrape --url https://aims-cameroon.org --resume
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose --profile tools run --rm indexer \
  python scripts/evaluate_rag.py --n 20
```

## Tests

```bash
docker compose -f docker-compose.test.yml run --rm --build tests   # 22 tests, throwaway Postgres, no API credits
docker compose -f docker-compose.test.yml down -v
```

## Deploy to AWS

A 3-tier setup: an AWS load balancer in front of a **web** server (Next.js), an **api** server (FastAPI) and a **db** server (Postgres), created with Terraform. See [deploy/README.md](deploy/README.md) for the one-time AWS account setup and costs, then:

```bash
./infra/tf.sh init && ./infra/tf.sh apply   # create the infrastructure (Terraform in Docker)
./deploy/deploy.sh ~/.ssh/aims-rag          # setup -> build -> test -> deploy -> smoke test
./infra/tf.sh destroy                       # remove everything to stop spending credits
```

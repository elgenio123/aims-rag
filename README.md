# AIMS Cameroon RAG Pipeline

An autonomous RAG (Retrieval-Augmented Generation) data engineering pipeline for building a high-quality, source-grounded knowledge base about AIMS Cameroon.

## Overview

This pipeline consists of five stages:
1. **Deep Web Scraping** - Crawl and extract content from AIMS Cameroon websites
2. **Document Storage** - Store structured documents with full metadata
3. **Chunking & Embedding** - Break documents into semantic chunks and embed them
4. **Vector Database Indexing** - Store chunks in FAISS/Chroma for efficient retrieval
5. **RAG Pipeline** - Answer questions using LangChain + Mistral LLM with strict source grounding

## Features

- ✅ Respects robots.txt and applies rate limiting
- ✅ Supports HTML pages and PDF documents
- ✅ Comprehensive metadata tracking and traceability
- ✅ Semantic chunking with context preservation
- ✅ Multiple vector database backends (FAISS, Chroma)
- ✅ Strict source grounding - no hallucinations
- ✅ Complete logging and traceability

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Clone the repository
2. Install uv (if you don't have it):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies (creates `.venv/` from `uv.lock`, using Python 3.12):
```bash
uv sync
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

## Usage

### 1. Scrape and Build Knowledge Base

```bash
uv run main.py scrape --url https://aims-cameroon.org
```

### 2. Query the Knowledge Base

```bash
uv run main.py query "What are the admission requirements for AIMS Cameroon?"
```

### 3. Full Pipeline (Scrape + Index + Query)

```bash
uv run main.py full --url https://aims-cameroon.org --query "Tell me about the academic programs"
```

### Managing Dependencies

```bash
uv add <package>        # add a dependency
uv remove <package>     # remove a dependency
uv lock --upgrade       # upgrade locked versions
```

## Project Structure

```
aims-rag/
├── config.py              # Configuration management
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions
├── main.py               # Main entry point
├── src/
│   ├── scraper/          # Web scraping module
│   ├── storage/          # Document storage system
│   ├── chunker/          # Chunking module
│   ├── embedder/         # Embedding and vector DB
│   ├── rag/              # RAG pipeline
│   └── utils/            # Utilities and logging
├── data/
│   ├── documents/        # Stored documents (JSON)
│   └── vector_db/        # Vector database files
└── logs/                 # Pipeline logs
```

## Knowledge Scope

The pipeline extracts information about:
- Institutional Information
- Academic Programs
- Admissions
- Funding & Scholarships
- Faculty & Research
- Student Life
- Administration & Contacts
- FAQs & Policies

## Architecture

### Stage 1: Web Scraping
- Polite crawling with rate limiting
- robots.txt compliance
- PDF and HTML parsing
- Duplicate detection

### Stage 2: Document Storage
- Structured JSON format
- Full metadata preservation
- Source URL tracking
- Timestamp recording

### Stage 3: Chunking
- 300-500 token chunks
- 50-100 token overlap
- Sentence boundary preservation
- Context retention

### Stage 4: Vector Database
- Sentence-transformers embeddings
- FAISS or Chroma storage
- Metadata preservation
- Cosine similarity search

### Stage 5: RAG Pipeline
- LangChain orchestration
- Mistral LLM generation
- Strict context grounding
- Source traceability

## Accuracy Policy

- ❌ No assumptions or inferred information
- ❌ No hallucinations
- ✅ Explicit "information not available" responses
- ✅ Every answer traceable to source

## License

AIMS License

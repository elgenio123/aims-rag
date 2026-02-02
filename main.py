"""Main CLI for AIMS Cameroon RAG Pipeline."""
import warnings

# Silence noisy deprecation warnings as early as possible
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from loguru import logger
from src.utils import setup_logging
from src.storage.storage import DocumentStorage
from src.chunker import chunk_document
from pathlib import Path
import json

setup_logging()

# Silence noisy deprecation warnings in console
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

def cmd_scrape(args):
    # Lazy import to avoid bringing heavy deps when not needed
    from src.scraper import Crawler
    urls = args.url
    crawler = Crawler(urls)
    docs = crawler.crawl()
    logger.info(f"Scraped and stored {len(docs)} documents.")


def _load_all_documents() -> list:
    storage = DocumentStorage()
    docs = []
    for p in storage.list_documents():
        with p.open('r', encoding='utf-8') as f:
            docs.append(json.load(f))
    return docs


def cmd_index(args):
    # Lazy import to avoid heavy dependencies during 'scrape'
    from src.rag import RagPipeline
    docs = _load_all_documents()
    # Chunk and build index
    all_chunks = []
    for d in docs:
        chunks = chunk_document(d)
        all_chunks.extend(chunks)
    logger.info(f"Produced {len(all_chunks)} chunks from {len(docs)} documents.")

    rag = RagPipeline()
    rag.build_index(all_chunks)


def cmd_full(args):
    # Lazy imports to avoid heavy dependencies during 'scrape'
    from src.scraper import Crawler
    from src.rag import RagPipeline
    cmd_scrape(args)
    cmd_index(args)
    cmd_query(args)


def cmd_query(args):
    # Lazy import to avoid heavy dependencies during 'scrape'
    from src.rag import RagPipeline
    question = args.query
    rag = RagPipeline()
    result = rag.answer(question)
    print("\n=== Answer ===\n")
    print(result['answer'])
    print("\n=== Trace ===\n")
    for t in result['trace']:
        print(f"doc_id={t['doc_id']} | source={t['source_url']} | score={t['score']}")


def build_parser():
    parser = argparse.ArgumentParser(description="AIMS Cameroon RAG Pipeline")

    sub = parser.add_subparsers(dest='command')

    p_scrape = sub.add_parser('scrape', help='Scrape starting URLs')
    p_scrape.add_argument('--url', nargs='+', required=True, help='Starting URLs to crawl')
    p_scrape.set_defaults(func=cmd_scrape)

    p_index = sub.add_parser('index', help='Chunk and build vector index')
    p_index.set_defaults(func=cmd_index)

    p_full = sub.add_parser('full', help='Scrape, index, and query')
    p_full.add_argument('--url', nargs='+', required=True, help='Starting URLs to crawl')
    p_full.add_argument('--query', required=True, help='Question to ask')
    p_full.set_defaults(func=cmd_full)

    p_query = sub.add_parser('query', help='Query the RAG pipeline')
    p_query.add_argument('query', help='Question to ask')
    p_query.set_defaults(func=cmd_query)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

#!/bin/bash
# Resume the AIMS Cameroon scrape after an interruption (e.g. a machine
# restart). Skips URLs that already have a stored document, so only the
# remaining unscraped pages are fetched. Safe to run any time, including
# after a full crawl already finished (it will just exit quickly).
set -euo pipefail

REPO_DIR="/home/elgenio/Documents/projects/aims-rag"
LOG_FILE="$REPO_DIR/logs/resume_scrape.log"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/logs"

{
  echo "=== Resume run started at $(date -Iseconds) ==="
  uv run main.py scrape --url https://aims-cameroon.org --resume
  echo "=== Resume run finished at $(date -Iseconds) ==="
} >> "$LOG_FILE" 2>&1

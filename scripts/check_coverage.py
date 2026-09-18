"""Verify that every canonical page in the site's sitemap has been scraped.

Usage:
    uv run python scripts/check_coverage.py [start_url]

Fetches the target site's sitemap (auto-discovered via robots.txt, falling
back to /wp-sitemap.xml and /sitemap.xml), and diffs it against the URLs
already present in data/documents/. Prints any sitemap URLs with no matching
stored document. Exits non-zero if any are missing (useful in CI/cron).
"""
import sys
import json
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper.crawler import Crawler
from config import DOCUMENTS_PATH


def main():
    start_url = sys.argv[1] if len(sys.argv) > 1 else "https://aims-cameroon.org"
    crawler = Crawler([start_url])
    sitemap_urls = {u.rstrip('/') for u in crawler._discover_sitemap_urls(start_url)}

    scraped_urls = set()
    for path in glob.glob(f"{DOCUMENTS_PATH}/*.json"):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        scraped_urls.add(doc["source_url"].rstrip('/'))

    missing = sorted(sitemap_urls - scraped_urls)

    print(f"Sitemap URLs:  {len(sitemap_urls)}")
    print(f"Scraped docs:  {len(scraped_urls)} (includes non-sitemap pages, e.g. /fr/ translations)")
    print(f"Missing:       {len(missing)}")
    if missing:
        print("\nSitemap URLs with no stored document:")
        for u in missing:
            print(f"  {u}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

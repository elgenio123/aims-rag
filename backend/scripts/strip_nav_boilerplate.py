"""Strip the repeated site nav-menu boilerplate from already-stored raw_text.

The crawler now strips this at scrape time (see src/utils/text_utils.py:
strip_nav_menu), but documents scraped before that fix still have it baked
into their raw_text on disk. This script retroactively cleans them.

Safe to re-run: documents that don't start with a known nav block (already
stripped, or PDFs, which never had one) are left untouched. Writes are
atomic (write to a .tmp file, then rename) so a concurrent crawl run writing
the same file can't leave a corrupted JSON file behind.

Usage:
    uv run python scripts/strip_nav_boilerplate.py           # dry run, report only
    uv run python scripts/strip_nav_boilerplate.py --apply   # write changes
"""
import sys
import json
import glob
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DOCUMENTS_PATH
from src.utils import strip_nav_menu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='write changes (default: dry run)')
    args = parser.parse_args()

    changed = 0
    unchanged = 0
    chars_removed = 0
    errors = []

    for path in sorted(glob.glob(f"{DOCUMENTS_PATH}/*.json")):
        try:
            with open(path, encoding='utf-8') as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append((path, str(e)))
            continue

        raw_text = doc.get('raw_text', '')
        new_text = strip_nav_menu(raw_text)
        if new_text == raw_text:
            unchanged += 1
            continue

        changed += 1
        chars_removed += len(raw_text) - len(new_text)

        if args.apply:
            doc['raw_text'] = new_text
            tmp_path = f"{path}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            Path(tmp_path).replace(path)

    mode = 'APPLIED' if args.apply else 'DRY RUN'
    print(f"[{mode}] {changed} documents had the nav block stripped")
    print(f"[{mode}] {unchanged} documents unchanged (no known nav block, e.g. PDFs)")
    print(f"[{mode}] {chars_removed:,} characters removed in total")
    if errors:
        print(f"[{mode}] {len(errors)} files could not be read:")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")
    if not args.apply and changed:
        print("\nRun again with --apply to write these changes.")


if __name__ == '__main__':
    main()

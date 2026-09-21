"""Resumable QA-pair generation pipeline for the AIMS admissions-assistant dataset.

The actual question/answer generation (fact extraction, persona/style writing,
grounding) is a judgment task done by an LLM (Claude) reading each document -
this script does NOT call an LLM. It only handles the mechanical parts that
must be exact and are cheap to verify in code:

  - deciding which documents are still unprocessed (resume support)
  - validating a generated batch against the source raw_text
      (every `evidence` string must be an exact substring; drop the pair if not)
  - enforcing the tier pair-cap as a safety net
  - dropping near-duplicate questions
  - assigning ids, writing <doc_id>.jsonl, and appending to _manifest.jsonl

Usage:
    # 1. See which documents still need processing (stable, sorted order)
    uv run python scripts/qa_pipeline.py next 10

    # 2. After generating QA pairs for those docs into a batch JSON file
    #    (see BATCH FORMAT below), validate + commit them:
    uv run python scripts/qa_pipeline.py commit path/to/batch.json

    # 3. Check overall progress
    uv run python scripts/qa_pipeline.py status

BATCH FORMAT (a JSON list, one object per document):
[
  {
    "doc_id": "...",
    "skipped": false,
    "reason": null,
    "facts_count": 7,
    "pairs": [
      {
        "question": "...",
        "answer": "...",
        "persona": "planning",
        "style": "casual",
        "question_type": "factual",
        "evidence": ["exact span from raw_text", "..."]
      },
      ...
    ]
  },
  {"doc_id": "...", "skipped": true, "reason": "no facts", "facts_count": 0, "pairs": []},
  ...
]
"""
import sys
import json
import glob
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DOCUMENTS_PATH

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_generated"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.jsonl"

VALID_PERSONAS = {"browsing", "planning", "applying", "admitted"}
VALID_STYLES = {"formal", "casual", "typos", "short", "multi"}
VALID_QTYPES = {"factual", "procedural", "yes_no", "multi_part"}

TIER_CAPS = {  # (min_chars, max_chars_exclusive, max_pairs)
    "thin": (0, 500, 8),
    "medium": (500, 2000, 15),
    "dense": (2000, float("inf"), 35),
}


def tier_for(chars: int) -> str:
    if chars < 500:
        return "thin"
    if chars < 2000:
        return "medium"
    return "dense"


def all_doc_ids_sorted():
    paths = sorted(glob.glob(f"{DOCUMENTS_PATH}/*.json"))
    return [Path(p).stem for p in paths]


def already_processed():
    done = set()
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def cmd_next(n: int):
    done = already_processed()
    remaining = [d for d in all_doc_ids_sorted() if d not in done]
    batch = remaining[:n]
    print(f"Total documents: {len(all_doc_ids_sorted())}")
    print(f"Already processed: {len(done)}")
    print(f"Remaining: {len(remaining)}")
    print(f"\nNext {len(batch)} doc_ids:")
    for d in batch:
        print(f"  {d}")


def cmd_status():
    done = already_processed()
    total = len(all_doc_ids_sorted())
    skipped = 0
    total_pairs = 0
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("skipped"):
                    skipped += 1
                total_pairs += rec.get("pairs", 0) or 0
    print(f"Total documents:   {total}")
    print(f"Processed:         {len(done)}")
    print(f"Remaining:         {total - len(done)}")
    print(f"Skipped (no facts/broken): {skipped}")
    print(f"Total QA pairs written:    {total_pairs}")


def normalize_question(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def load_source_doc(doc_id: str):
    path = Path(DOCUMENTS_PATH) / f"{doc_id}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_and_write(doc_result: dict, log):
    doc_id = doc_result.get("doc_id")
    if not doc_id:
        log.append("batch entry missing doc_id, skipped entirely")
        return None

    try:
        src = load_source_doc(doc_id)
    except (OSError, json.JSONDecodeError) as e:
        log.append(f"{doc_id}: could not load source document ({e}) - marking skipped/broken")
        write_empty(doc_id)
        return {"doc_id": doc_id, "chars": 0, "tier": "thin", "facts": 0,
                "pairs": 0, "skipped": True, "reason": f"source unreadable: {e}"}

    raw_text = src.get("raw_text", "") or ""
    chars = len(raw_text)
    tier = tier_for(chars)
    _, _, cap = TIER_CAPS[tier]

    if doc_result.get("skipped"):
        write_empty(doc_id)
        reason = doc_result.get("reason") or "skipped"
        return {"doc_id": doc_id, "chars": chars, "tier": tier, "facts": doc_result.get("facts_count", 0),
                "pairs": 0, "skipped": True, "reason": reason}

    raw_pairs = doc_result.get("pairs", [])
    accepted = []
    seen_norms = set()

    for i, p in enumerate(raw_pairs):
        reason = None
        question = (p.get("question") or "").strip()
        answer = (p.get("answer") or "").strip()
        persona = p.get("persona")
        style = p.get("style")
        qtype = p.get("question_type")
        evidence = p.get("evidence") or []

        if not question or not answer:
            reason = "missing question/answer"
        elif persona not in VALID_PERSONAS:
            reason = f"invalid persona {persona!r}"
        elif style not in VALID_STYLES:
            reason = f"invalid style {style!r}"
        elif qtype not in VALID_QTYPES:
            reason = f"invalid question_type {qtype!r}"
        elif not evidence or not isinstance(evidence, list):
            reason = "no evidence spans"
        else:
            bad_span = next((e for e in evidence if not isinstance(e, str) or e not in raw_text), None)
            if bad_span is not None:
                reason = f"evidence not an exact substring: {bad_span[:80]!r}"

        if reason:
            log.append(f"{doc_id} pair #{i}: DROPPED - {reason}")
            continue

        norm = normalize_question(question)
        if norm in seen_norms:
            log.append(f"{doc_id} pair #{i}: DROPPED - near-duplicate question")
            continue
        seen_norms.add(norm)

        accepted.append({
            "question": question, "answer": answer, "persona": persona,
            "style": style, "question_type": qtype, "evidence": evidence,
        })

    if len(accepted) > cap:
        log.append(f"{doc_id}: {len(accepted)} valid pairs exceeds {tier} cap of {cap}; trimming")
        accepted = accepted[:cap]

    out_path = OUTPUT_DIR / f"{doc_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(accepted, start=1):
            line = {
                "id": f"{doc_id}_{i:03d}",
                "doc_id": doc_id,
                "question": p["question"],
                "answer": p["answer"],
                "persona": p["persona"],
                "style": p["style"],
                "question_type": p["question_type"],
                "evidence": p["evidence"],
                "source_url": src.get("source_url", ""),
                "category": src.get("category", ""),
                "scrape_timestamp": src.get("scrape_timestamp", ""),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    skipped_final = len(accepted) == 0
    return {
        "doc_id": doc_id, "chars": chars, "tier": tier,
        "facts": doc_result.get("facts_count", 0), "pairs": len(accepted),
        "skipped": skipped_final,
        "reason": ("no valid pairs survived validation" if skipped_final else None),
    }


def write_empty(doc_id: str):
    (OUTPUT_DIR / f"{doc_id}.jsonl").write_text("", encoding="utf-8")


def cmd_commit(batch_path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(batch_path, encoding="utf-8") as f:
        batch = json.load(f)

    log = []
    manifest_records = []
    for doc_result in batch:
        rec = validate_and_write(doc_result, log)
        if rec:
            manifest_records.append(rec)

    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Committed {len(manifest_records)} documents.")
    total_pairs = sum(r["pairs"] for r in manifest_records)
    print(f"Total valid QA pairs written this batch: {total_pairs}")
    if log:
        print(f"\n{len(log)} validation notes:")
        for line in log:
            print(f"  - {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    cmd = sys.argv[1]
    if cmd == "next":
        cmd_next(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "status":
        cmd_status()
    elif cmd == "commit":
        cmd_commit(sys.argv[2])
    else:
        print(__doc__)
        raise SystemExit(1)

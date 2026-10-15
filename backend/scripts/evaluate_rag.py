"""Evaluate RagPipeline.answer() against ground-truth QA pairs.

Samples N question/answer pairs from data/qa_generated/*.jsonl, asks the live
RAG pipeline each question, and scores the generated answer against the
ground-truth answer with:

  - Exact match (normalized)
  - Token-level Precision / Recall / F1 (SQuAD-style)
  - ROUGE-1 / ROUGE-2 / ROUGE-L (F-measure)
  - BLEU (sacrebleu, sentence-level)
  - Embedding cosine similarity (reuses the pipeline's own embedding backend)
  - Retrieval hit rate (whether the ground-truth doc_id was among the
    retrieved chunks) -- a RAG-specific metric distinct from answer quality

Every LLM call costs money and hits a real API (OPENROUTER_API_KEY /
MISTRAL_API_KEY, per config.py) -- this is not a dry-run tool.

Usage:
    uv run python scripts/evaluate_rag.py                  # 100 random questions
    uv run python scripts/evaluate_rag.py --n 20 --seed 7   # smaller/cheaper run
    uv run python scripts/evaluate_rag.py --top-k 6

Outputs:
    data/eval/results_<timestamp>.jsonl   -- per-example scores + raw answers
    Summary table printed to stdout.
"""
import argparse
import json
import random
import re
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QA_DIR = ROOT / "data" / "qa_generated"
EVAL_DIR = ROOT / "data" / "eval"


def load_pairs():
    pairs = []
    for path in sorted(QA_DIR.glob("*.jsonl")):
        if path.name == "_manifest.jsonl":
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs.append(json.loads(line))
    return pairs


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def token_f1(pred: str, ref: str):
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        eq = pred_tokens == ref_tokens
        return (1.0, 1.0, 1.0) if eq else (0.0, 0.0, 0.0)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return (0.0, 0.0, 0.0)
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def embed_text(indexer, text: str):
    if indexer.backend_name == "fastembed":
        v = next(indexer.fe.embed([text]))
        return v.tolist() if hasattr(v, "tolist") else list(v)
    v = indexer.model.encode([text], normalize_embeddings=True)[0]
    return v.tolist() if hasattr(v, "tolist") else list(v)


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of questions to sample (default: 100)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling (default: 42)")
    ap.add_argument("--top-k", type=int, default=4, help="top_k passed to RagPipeline.answer() (default: 4)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between LLM calls (default: 0)")
    args = ap.parse_args()

    pairs = load_pairs()
    if not pairs:
        print(f"No QA pairs found under {QA_DIR}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    n = min(args.n, len(pairs))
    sample = rng.sample(pairs, n)
    print(f"Loaded {len(pairs)} QA pairs; evaluating on {n} (seed={args.seed}, top_k={args.top_k})")

    from rouge_score import rouge_scorer
    import sacrebleu
    from src.rag.pipeline import RagPipeline

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    pipeline = RagPipeline()

    results = []
    for i, pair in enumerate(sample, 1):
        question = pair["question"]
        gold_answer = pair["answer"]
        gold_doc_id = pair.get("doc_id")

        try:
            out = pipeline.answer(question, top_k=args.top_k)
        except Exception as e:
            print(f"[{i}/{n}] ERROR calling pipeline for question {question!r}: {e}", file=sys.stderr)
            results.append({
                "question": question, "gold_answer": gold_answer, "doc_id": gold_doc_id,
                "predicted_answer": None, "error": str(e),
            })
            continue

        pred_answer = out["answer"].strip()
        trace = out.get("trace", [])
        retrieved_doc_ids = [t.get("doc_id") for t in trace]
        retrieval_hit = 1.0 if gold_doc_id in retrieved_doc_ids else 0.0

        em = exact_match(pred_answer, gold_answer)
        precision, recall, f1 = token_f1(pred_answer, gold_answer)
        rouge = scorer.score(gold_answer, pred_answer)
        bleu = sacrebleu.sentence_bleu(pred_answer, [gold_answer]).score

        try:
            emb_pred = embed_text(pipeline.indexer, pred_answer)
            emb_gold = embed_text(pipeline.indexer, gold_answer)
            cos_sim = cosine(emb_pred, emb_gold)
        except Exception:
            cos_sim = None

        refused = pred_answer.strip().lower().startswith("the information is not available")

        row = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": pred_answer,
            "doc_id": gold_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_hit": retrieval_hit,
            "refused": refused,
            "exact_match": em,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "rouge1_f": rouge["rouge1"].fmeasure,
            "rouge2_f": rouge["rouge2"].fmeasure,
            "rougeL_f": rouge["rougeL"].fmeasure,
            "bleu": bleu,
            "embedding_cosine": cos_sim,
        }
        results.append(row)
        print(f"[{i}/{n}] EM={em:.0f} F1={f1:.2f} ROUGE-L={rouge['rougeL'].fmeasure:.2f} "
              f"BLEU={bleu:.1f} hit={retrieval_hit:.0f}  Q: {question[:70]}")

        if args.sleep:
            time.sleep(args.sleep)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = EVAL_DIR / f"results_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = [r for r in results if "error" not in r]
    n_errors = len(results) - len(scored)

    print("\n" + "=" * 60)
    print(f"RAG EVALUATION SUMMARY  (n={len(results)}, errors={n_errors})")
    print("=" * 60)
    print(f"Exact Match:          {mean([r['exact_match'] for r in scored]):.3f}")
    print(f"Token Precision:      {mean([r['token_precision'] for r in scored]):.3f}")
    print(f"Token Recall:         {mean([r['token_recall'] for r in scored]):.3f}")
    print(f"Token F1:             {mean([r['token_f1'] for r in scored]):.3f}")
    print(f"ROUGE-1 F:            {mean([r['rouge1_f'] for r in scored]):.3f}")
    print(f"ROUGE-2 F:            {mean([r['rouge2_f'] for r in scored]):.3f}")
    print(f"ROUGE-L F:            {mean([r['rougeL_f'] for r in scored]):.3f}")
    print(f"BLEU:                 {mean([r['bleu'] for r in scored]):.2f}")
    print(f"Embedding Cosine Sim: {mean([r['embedding_cosine'] for r in scored]):.3f}")
    print(f"Retrieval Hit Rate:   {mean([r['retrieval_hit'] for r in scored]):.3f}")
    print(f"Refusal Rate:         {mean([1.0 if r['refused'] else 0.0 for r in scored]):.3f}")
    print("=" * 60)
    print(f"Per-example results written to {out_path}")


if __name__ == "__main__":
    main()

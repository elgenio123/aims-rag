"""Convert data/qa_generated/*.jsonl into Together AI chat-format fine-tuning files.

Each training example mirrors the exact prompt RagPipeline.answer() builds at
inference time (src/rag/pipeline.py): a system message with the grounding
policy, and a user message of "Context:\n[doc_id=... | category=... |
source=...]\n{evidence}\n\nQuestion: {question}\n\nAnswer:". The evidence
quotes attached to each QA pair stand in for the retrieved chunk text.

A ratio of synthetic refusal examples is mixed in so the fine-tuned model
retains the "not available in the current documents" behavior, since none of
the real QA pairs are refusals.

Usage: uv run python scripts/prepare_finetune_data.py
Outputs: data/finetune/train.jsonl, data/finetune/val.jsonl
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.pipeline import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE
QA_DIR = ROOT / "data" / "qa_generated"
OUT_DIR = ROOT / "data" / "finetune"
REFUSAL_TEXT = "The information is not available in the current documents."
SEED = 42
VAL_FRACTION = 0.10
MISMATCHED_REFUSAL_COUNT = 150
OOD_CONTEXTS_PER_QUESTION = 3

OOD_QUESTIONS = [
    "What's the weather like in Paris today?",
    "How do I install Python on Windows?",
    "What is the recipe for jollof rice?",
    "Who won the 2022 FIFA World Cup?",
    "What's the boiling point of water at sea level?",
    "How do I reset my Gmail password?",
    "What time zone is Tokyo in?",
    "Can you recommend a good sci-fi movie?",
    "What's the exchange rate between USD and EUR?",
    "How many bones are in the human body?",
    "What is the tallest mountain in the world?",
    "How do I change a flat tire?",
    "What's the difference between a virus and bacteria?",
    "Who wrote Romeo and Juliet?",
    "What's the population of Brazil?",
    "How do I train for a marathon?",
    "What are the symptoms of the common cold?",
    "What is quantum entanglement in simple terms?",
    "How do I apply for a US visa?",
    "What's a good beginner guitar?",
    "Where can I find cheap flights to London?",
    "How do I bake sourdough bread?",
    "What's the capital of Australia?",
    "How does compound interest work?",
]


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


def format_context(doc_id, category, source_url, evidence):
    text = "\n".join(evidence)
    return f"[doc_id={doc_id} | category={category} | source={source_url}]\n{text}"


def build_messages(question, context, answer):
    prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} real QA pairs from {QA_DIR}")

    rng = random.Random(SEED)
    examples = []

    # Real grounded pairs
    for p in pairs:
        context = format_context(p["doc_id"], p.get("category", ""), p.get("source_url", ""), p["evidence"])
        examples.append(build_messages(p["question"], context, p["answer"]))

    # Synthetic refusals: real question + a different, unrelated doc's context
    for _ in range(MISMATCHED_REFUSAL_COUNT):
        q_pair = rng.choice(pairs)
        candidates = [c for c in pairs if c["doc_id"] != q_pair["doc_id"] and c.get("category") != q_pair.get("category")]
        if not candidates:
            candidates = [c for c in pairs if c["doc_id"] != q_pair["doc_id"]]
        wrong_ctx_pair = rng.choice(candidates)
        context = format_context(
            wrong_ctx_pair["doc_id"], wrong_ctx_pair.get("category", ""),
            wrong_ctx_pair.get("source_url", ""), wrong_ctx_pair["evidence"],
        )
        examples.append(build_messages(q_pair["question"], context, REFUSAL_TEXT))

    # Synthetic refusals: out-of-domain question + random real context
    for q in OOD_QUESTIONS:
        for _ in range(OOD_CONTEXTS_PER_QUESTION):
            ctx_pair = rng.choice(pairs)
            context = format_context(
                ctx_pair["doc_id"], ctx_pair.get("category", ""),
                ctx_pair.get("source_url", ""), ctx_pair["evidence"],
            )
            examples.append(build_messages(q, context, REFUSAL_TEXT))

    n_refusals = MISMATCHED_REFUSAL_COUNT + len(OOD_QUESTIONS) * OOD_CONTEXTS_PER_QUESTION
    print(f"Added {n_refusals} synthetic refusal examples ({n_refusals / len(examples):.1%} of total)")

    # Validate schema: alternating roles, ends in assistant
    for ex in examples:
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"], f"bad roles: {roles}"
        for m in ex["messages"]:
            assert isinstance(m["content"], str) and m["content"].strip(), "empty content"

    rng.shuffle(examples)
    n_val = int(len(examples) * VAL_FRACTION)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train_examples)} -> {train_path}")
    print(f"Val:   {len(val_examples)} -> {val_path}")


if __name__ == "__main__":
    main()

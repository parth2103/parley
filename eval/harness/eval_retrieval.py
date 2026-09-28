"""Evaluate BM25 + hashed n-gram retriever on known-answer policy questions.

Measures exact Recall@1, Recall@3, and Recall@5 on:
1. Canonical known-answer questions (n=10)
2. Grown golden evaluation dataset (n=29 answerable items) with lexical gaps and conversational phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rag.retriever import BM25HashedNgramRetriever, HybridRetriever
from eval.scenarios.golden_set import GOLDEN_SET


@dataclass(frozen=True)
class KnownAnswerEvalItem:
    query_id: str
    query: str
    target_chunk_id: str
    trap_chunk_ids: list[str]
    description: str


EVAL_ITEMS: list[KnownAnswerEvalItem] = [
    KnownAnswerEvalItem(
        query_id="QA-01",
        query="What is the standard comprehensive deductible on auto policy POL-4401?",
        target_chunk_id="CHUNK-AUTO-01",
        trap_chunk_ids=["CHUNK-AUTO-02", "CHUNK-AUTO-03"],
        description="Must retrieve $500 standard deductible, not $100 glass or $50 towing.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-02",
        query="How much is the deductible for repairing a chipped windshield?",
        target_chunk_id="CHUNK-AUTO-02",
        trap_chunk_ids=["CHUNK-AUTO-01"],
        description="Must retrieve glass endorsement with $100 deductible and $0 chip repair.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-03",
        query="What is the maximum coverage and deductible for emergency towing assistance?",
        target_chunk_id="CHUNK-AUTO-03",
        trap_chunk_ids=["CHUNK-AUTO-01"],
        description="Must retrieve $150 towing limit and $50 deductible.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-04",
        query="What are the daily and total limits for rental car reimbursement?",
        target_chunk_id="CHUNK-AUTO-04",
        trap_chunk_ids=[],
        description="Must retrieve $45/day and $1,350 total limit.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-05",
        query="Is a burst frozen pipe covered, and what is the maximum coverage limit?",
        target_chunk_id="CHUNK-HOME-01",
        trap_chunk_ids=["CHUNK-HOME-02", "CHUNK-HOME-03"],
        description="Must retrieve $25,000 plumbing freeze limit, not $5,000 sewer backup or $0 flood.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-06",
        query="Does the policy cover sump pump failure or sewer water backup?",
        target_chunk_id="CHUNK-HOME-02",
        trap_chunk_ids=["CHUNK-HOME-01", "CHUNK-HOME-03"],
        description="Must retrieve $5,000 water backup rider with $1,000 deductible.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-07",
        query="Is groundwater or surface flooding covered under standard homeowners?",
        target_chunk_id="CHUNK-HOME-03",
        trap_chunk_ids=["CHUNK-HOME-01", "CHUNK-HOME-02"],
        description="Must retrieve Exclusion F ($0 limit, flood excluded).",
    ),
    KnownAnswerEvalItem(
        query_id="QA-08",
        query="What is the coverage limit if personal items are stolen from my locked vehicle?",
        target_chunk_id="CHUNK-PROP-01",
        trap_chunk_ids=["CHUNK-PROP-02", "CHUNK-PROP-03"],
        description="Must retrieve $1,500 off-premises vehicle theft limit, not $1,000 jewelry sub-limit.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-09",
        query="Within how many hours must a theft or vandalism police report be filed?",
        target_chunk_id="CHUNK-CLAIM-02",
        trap_chunk_ids=["CHUNK-CLAIM-01"],
        description="Must retrieve 24-hour mandatory police report, not 30-day general notice.",
    ),
    KnownAnswerEvalItem(
        query_id="QA-10",
        query="What is the percentage threshold of actual cash value to declare a vehicle a total loss?",
        target_chunk_id="CHUNK-CLAIM-04",
        trap_chunk_ids=[],
        description="Must retrieve 75 percent constructive total loss valuation threshold.",
    ),
]


def run_eval():
    retriever = BM25HashedNgramRetriever()
    print("=" * 70)
    print("BM25 + HASHED N-GRAM RETRIEVER EVALUATION")
    print("=" * 70)

    # 1. Canonical known-answer items
    print(f"\n--- Part 1: Canonical Known-Answer Set (n={len(EVAL_ITEMS)}) ---")
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    for item in EVAL_ITEMS:
        results_5 = retriever.retrieve(item.query, top_k=5)
        top_ids_5 = [chunk.chunk_id for chunk, _ in results_5]
        top_ids_3 = top_ids_5[:3]
        top_id_1 = top_ids_5[0] if top_ids_5 else ""

        hit_1 = item.target_chunk_id == top_id_1
        hit_3 = item.target_chunk_id in top_ids_3
        hit_5 = item.target_chunk_id in top_ids_5

        if hit_1:
            hits_at_1 += 1
        if hit_3:
            hits_at_3 += 1
        if hit_5:
            hits_at_5 += 1

        rank = (top_ids_5.index(item.target_chunk_id) + 1) if hit_5 else ">5"
        status = "PASS (top-1)" if hit_1 else ("PASS (top-3)" if hit_3 else ("PASS (top-5)" if hit_5 else "FAIL"))
        print(f"[{item.query_id}] Target: {item.target_chunk_id} | Rank: {rank} | Status: {status}")

    canon_r1 = hits_at_1 / len(EVAL_ITEMS)
    canon_r3 = hits_at_3 / len(EVAL_ITEMS)
    canon_r5 = hits_at_5 / len(EVAL_ITEMS)
    print(f"Canonical Recall@1: {canon_r1:.4f} ({hits_at_1}/{len(EVAL_ITEMS)})")
    print(f"Canonical Recall@3: {canon_r3:.4f} ({hits_at_3}/{len(EVAL_ITEMS)})")
    print(f"Canonical Recall@5: {canon_r5:.4f} ({hits_at_5}/{len(EVAL_ITEMS)})")

    # 2. Grown Golden Set (all answerable items)
    answerable_items = [item for item in GOLDEN_SET if item.is_answerable]
    print(f"\n--- Part 2: Grown Golden Evaluation Set (n={len(answerable_items)} answerable items) ---")
    g_hits_at_1 = 0
    g_hits_at_3 = 0
    g_hits_at_5 = 0

    misses_at_3: list[tuple[str, str, list[str], list[str]]] = []

    for item in answerable_items:
        results_5 = retriever.retrieve(item.question, top_k=5)
        top_ids_5 = [chunk.chunk_id for chunk, _ in results_5]
        top_ids_3 = top_ids_5[:3]
        top_id_1 = top_ids_5[0] if top_ids_5 else ""

        hit_1 = any(t == top_id_1 for t in item.target_chunk_ids)
        hit_3 = any(t in top_ids_3 for t in item.target_chunk_ids)
        hit_5 = any(t in top_ids_5 for t in item.target_chunk_ids)

        if hit_1:
            g_hits_at_1 += 1
        if hit_3:
            g_hits_at_3 += 1
        if hit_5:
            g_hits_at_5 += 1

        if not hit_3:
            misses_at_3.append((item.id, item.question, item.target_chunk_ids, top_ids_3))

    g_r1 = g_hits_at_1 / len(answerable_items)
    g_r3 = g_hits_at_3 / len(answerable_items)
    g_r5 = g_hits_at_5 / len(answerable_items)

    print(f"Grown Golden Set Recall@1: {g_r1:.4f} ({g_hits_at_1}/{len(answerable_items)})")
    print(f"Grown Golden Set Recall@3: {g_r3:.4f} ({g_hits_at_3}/{len(answerable_items)})")
    print(f"Grown Golden Set Recall@5: {g_r5:.4f} ({g_hits_at_5}/{len(answerable_items)})")

    if misses_at_3:
        print(f"\nItems missing top-3 (demonstrating eval is discriminative / off ceiling):")
        for qid, qtext, targets, top3 in misses_at_3:
            print(f"  [{qid}] Targets: {targets} | Top-3: {top3}")
            print(f"       Query: {qtext}")

    print("=" * 70)
    return {
        "canonical": {"r1": canon_r1, "r3": canon_r3, "r5": canon_r5},
        "grown_golden_set": {"n": len(answerable_items), "r1": g_r1, "r3": g_r3, "r5": g_r5},
    }


if __name__ == "__main__":
    run_eval()


"""CI Smoke Evaluation Script for Continuous Integration Gate.

Runs a fast text-mode smoke evaluation on a 10-item subset of the Golden Set (k=1)
using Groq (openai/gpt-oss-120b).

Gate conditions:
1. Task correctness >= baseline (0.7356) - tolerance (0.05) = 0.6856 (i.e. >= 75% on answerable items).
2. Groundedness >= 0.90 (>= 90% with zero hallucinated figures).

Exits with code 0 on pass, code 1 on regression.
Estimated cost per run: ~$0.0014 (approx 0.14 cents).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
import openai

from backend.pipeline.config import configure_tls
from backend.rag.retriever import HybridRetriever
from eval.scenarios.golden_set import GOLDEN_SET, GoldenItem
from eval.harness.discriminative_paired_eval import (
    DiscriminativeScorer,
    SYSTEM_PROMPT,
    format_context_prompt,
)

SMOKE_ITEM_IDS = [
    # 8 Answerable items representing baseline distribution (Auto, Home, Claims, Multi-chunk, Noisy ASR)
    "GS-01",  # Auto comprehensive animal strike ($500)
    "GS-04",  # Auto windshield glass replacement ($100)
    "GS-06",  # Auto roadside towing ($100 limit)
    "GS-10",  # Homeowners dwelling coverage ($350,000)
    "GS-18",  # Claims prompt reporting deadline (60 days)
    "GS-28",  # Multi-chunk homeowners freeze + personal property limit ($25,000)
    "GS-30",  # Noisy ASR property theft claim ($3,000 electronics rider)
    "GS-27",  # Multi-chunk auto total loss (harder case)
    # 2 Unanswerable items requiring clean refusal
    "GS-22",  # Pet veterinary illness exclusion
    "GS-23",  # Commercial food delivery exclusion
]

BASELINE_TASK_CORRECTNESS = 0.7356
CORRECTNESS_TOLERANCE = 0.05
MIN_CORRECTNESS_THRESHOLD = BASELINE_TASK_CORRECTNESS - CORRECTNESS_TOLERANCE  # 0.6856
MIN_GROUNDEDNESS_THRESHOLD = 0.90


async def run_ci_smoke_eval() -> int:
    load_dotenv()
    configure_tls()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is required for CI smoke eval.")
        return 1

    client = openai.AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"
    retriever = HybridRetriever()

    items_by_id = {it.id: it for it in GOLDEN_SET}
    smoke_items = [items_by_id[iid] for iid in SMOKE_ITEM_IDS if iid in items_by_id]

    print("=" * 70)
    print(f"PARLEY CI SMOKE EVALUATION GATE (n={len(smoke_items)} items, k=1)")
    print(f"Model: {model} | Baseline: {BASELINE_TASK_CORRECTNESS:.4f} (Margin: {CORRECTNESS_TOLERANCE})")
    print("=" * 70)

    task_correct_count = 0
    grounded_count = 0
    answerable_count = 0
    refusal_accurate_count = 0
    unanswerable_count = 0

    for item in smoke_items:
        retrieved_results = retriever.retrieve(item.question, top_k=3)
        retrieved_chunks = [chunk for chunk, _ in retrieved_results]
        user_prompt = format_context_prompt(item.question, retrieved_chunks)

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        response_text = resp.choices[0].message.content or ""

        (
            is_task_correct,
            is_grounded,
            hallucinated_nums,
            is_refused,
            refusal_accurate,
        ) = DiscriminativeScorer.score_response(item, response_text, retrieved_chunks)

        if item.is_answerable:
            answerable_count += 1
            if is_task_correct:
                task_correct_count += 1
        else:
            unanswerable_count += 1
            if refusal_accurate:
                refusal_accurate_count += 1

        if is_grounded:
            grounded_count += 1

        status_str = "PASS" if is_task_correct else "FAIL"
        print(f"[{item.id}] ({'Ans' if item.is_answerable else 'Unans'}) {status_str} | Grounded: {is_grounded} | Q: {item.question[:55]}...")
        if not is_task_correct:
            print(f"       Response: {response_text[:90]}...")
            if hallucinated_nums:
                print(f"       Hallucinated: {hallucinated_nums}")

    task_corr_score = task_correct_count / answerable_count if answerable_count > 0 else 0.0
    grounded_score = grounded_count / len(smoke_items)
    refusal_score = refusal_accurate_count / unanswerable_count if unanswerable_count > 0 else 1.0

    print("\n" + "=" * 70)
    print("CI GATE VERIFICATION REPORT")
    print("=" * 70)
    print(f"Answerable Items Evaluated: {answerable_count}")
    print(f"Unanswerable Items Evaluated: {unanswerable_count}")
    print(f"Task Correctness: {task_corr_score:.4f} (Threshold: >= {MIN_CORRECTNESS_THRESHOLD:.4f})")
    print(f"Groundedness:     {grounded_score:.4f} (Threshold: >= {MIN_GROUNDEDNESS_THRESHOLD:.4f})")
    print(f"Refusal Accuracy: {refusal_score:.4f}")
    print("-" * 70)

    # Gate verification
    failed_reasons = []
    if task_corr_score < MIN_CORRECTNESS_THRESHOLD:
        failed_reasons.append(
            f"Task correctness {task_corr_score:.4f} dropped below minimum threshold {MIN_CORRECTNESS_THRESHOLD:.4f}"
        )
    if grounded_score < MIN_GROUNDEDNESS_THRESHOLD:
        failed_reasons.append(
            f"Groundedness {grounded_score:.4f} dropped below minimum threshold {MIN_GROUNDEDNESS_THRESHOLD:.4f}"
        )

    if failed_reasons:
        print("CI GATE STATUS: FAILED")
        for r in failed_reasons:
            print(f"  [X] {r}")
        print("=" * 70)
        return 1
    else:
        print("CI GATE STATUS: PASSED")
        print("  [✓] All regression thresholds satisfied.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_ci_smoke_eval())
    sys.exit(exit_code)

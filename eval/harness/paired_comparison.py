"""Paired LLM Comparison on Golden Set (Text Mode): Groq vs Claude.

Measures:
1. TTFT (p50, p95) and Total Latency (p50, p95) with paired differences and 95% CIs.
2. Correctness, Groundedness (hallucinations on numbers/dates), and Refusal Accuracy.
3. Token usage and Cost per 1,000 queries at published rates.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anthropic
import numpy as np
import openai
from dotenv import load_dotenv
from scipy import stats

from backend.rag.corpus import CORPUS, PolicyChunk
from backend.rag.retriever import HybridRetriever
from eval.harness.text_eval import (
    SYSTEM_PROMPT,
    DeterministicScorer,
    format_context_prompt,
)
from eval.scenarios.golden_set import GOLDEN_SET, GoldenItem

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"

# Published rates per 1,000,000 tokens
PRICING = {
    "claude-sonnet-4-6": {"input_per_m": 3.00, "output_per_m": 15.00},
    "openai/gpt-oss-120b": {"input_per_m": 0.15, "output_per_m": 0.60},
}


@dataclass
class PairedTurnResult:
    item_id: str
    category: str
    question: str
    is_answerable: bool
    retrieved_chunk_ids: list[str]
    # Groq metrics
    groq_response: str
    groq_ttft_s: float
    groq_total_s: float
    groq_in_tokens: int
    groq_out_tokens: int
    groq_is_correct: bool
    groq_is_grounded: bool
    groq_hallucinated_numbers: list[str]
    groq_is_refused: bool
    groq_refusal_accurate: bool
    # Claude metrics
    claude_response: str
    claude_ttft_s: float
    claude_total_s: float
    claude_in_tokens: int
    claude_out_tokens: int
    claude_is_correct: bool
    claude_is_grounded: bool
    claude_hallucinated_numbers: list[str]
    claude_is_refused: bool
    claude_refusal_accurate: bool


async def query_claude(client: anthropic.AsyncAnthropic, prompt: str) -> tuple[float, float, str, int, int]:
    t0 = time.perf_counter()
    stream = await client.messages.create(
        model=ANTHROPIC_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
        stream=True,
    )
    first_token_time = None
    chunks = []
    in_tok = 0
    out_tok = 0
    async for event in stream:
        if event.type == "message_start":
            in_tok = event.message.usage.input_tokens
        elif event.type == "content_block_delta" and hasattr(event.delta, "text"):
            if event.delta.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t0
                chunks.append(event.delta.text)
        elif event.type == "message_delta":
            out_tok = event.usage.output_tokens

    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    return ttft, total_time, "".join(chunks).strip(), in_tok, out_tok


async def query_groq(client: openai.AsyncOpenAI, prompt: str) -> tuple[float, float, str, int, int]:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    first_token_time = None
    chunks = []
    in_tok = 0
    out_tok = 0
    async for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            in_tok = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens

    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    return ttft, total_time, "".join(chunks).strip(), in_tok, out_tok


def bootstrap_ci(diffs: list[float], num_samples: int = 5000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval of the mean difference."""
    data = np.array(diffs)
    n = len(data)
    boot_means = []
    rng = np.random.default_rng(seed=42)
    for _ in range(num_samples):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(data)), low, high


def calc_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    if len(values) == 1:
        return {"p50": values[0], "p95": values[0]}
    cuts = quantiles(values, n=100, method="inclusive")
    return {"p50": median(values), "p95": cuts[94]}


async def run_paired_benchmark() -> dict[str, Any]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not anthropic_key or not groq_key:
        raise ValueError("Missing ANTHROPIC_API_KEY or GROQ_API_KEY")

    claude_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
    groq_client = openai.AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    retriever = HybridRetriever(CORPUS)
    scorer = DeterministicScorer()

    print(f"--- PARLEY STEP 4 PAIRED LLM BENCHMARK ---")
    print(f"Groq Model:   {GROQ_MODEL}")
    print(f"Claude Model: {ANTHROPIC_MODEL}")
    print(f"Items:        n={len(GOLDEN_SET)} (21 answerable, 5 unanswerable)")
    print(f"Scorer:       Deterministic (facts/numbers/dates/refusals)")
    print("Performing client warmup...")
    await query_claude(claude_client, "Warmup ping.")
    await query_groq(groq_client, "Warmup ping.")
    print("Warmup complete. Starting interleaved paired evaluation...\n")

    paired_results: list[PairedTurnResult] = []

    for idx, item in enumerate(GOLDEN_SET, start=1):
        retrieved_pairs = retriever.retrieve(item.question, top_k=3)
        retrieved_chunks = [chunk for chunk, _score in retrieved_pairs]
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        prompt = format_context_prompt(item.question, retrieved_chunks)

        # Alternating order to eliminate order bias
        if idx % 2 == 1:
            g_ttft, g_total, g_resp, g_in, g_out = await query_groq(groq_client, prompt)
            c_ttft, c_total, c_resp, c_in, c_out = await query_claude(claude_client, prompt)
        else:
            c_ttft, c_total, c_resp, c_in, c_out = await query_claude(claude_client, prompt)
            g_ttft, g_total, g_resp, g_in, g_out = await query_groq(groq_client, prompt)

        g_corr, g_grnd, g_halls, g_ref, g_ref_acc = scorer.score_item(item, g_resp, retrieved_chunks)
        c_corr, c_grnd, c_halls, c_ref, c_ref_acc = scorer.score_item(item, c_resp, retrieved_chunks)

        turn = PairedTurnResult(
            item_id=item.id,
            category=item.category,
            question=item.question,
            is_answerable=item.is_answerable,
            retrieved_chunk_ids=retrieved_ids,
            groq_response=g_resp,
            groq_ttft_s=g_ttft,
            groq_total_s=g_total,
            groq_in_tokens=g_in,
            groq_out_tokens=g_out,
            groq_is_correct=g_corr,
            groq_is_grounded=g_grnd,
            groq_hallucinated_numbers=g_halls,
            groq_is_refused=g_ref,
            groq_refusal_accurate=g_ref_acc,
            claude_response=c_resp,
            claude_ttft_s=c_ttft,
            claude_total_s=c_total,
            claude_in_tokens=c_in,
            claude_out_tokens=c_out,
            claude_is_correct=c_corr,
            claude_is_grounded=c_grnd,
            claude_hallucinated_numbers=c_halls,
            claude_is_refused=c_ref,
            claude_refusal_accurate=c_ref_acc,
        )
        paired_results.append(turn)

        print(
            f"[{idx:02d}/{len(GOLDEN_SET):02d}] {item.id} | "
            f"Groq TTFT: {g_ttft:.3f}s (Corr={int(g_corr)}, Grnd={int(g_grnd)}) | "
            f"Claude TTFT: {c_ttft:.3f}s (Corr={int(c_corr)}, Grnd={int(c_grnd)})"
        )

    # Statistical Aggregations
    n = len(paired_results)
    g_ttfts = [r.groq_ttft_s for r in paired_results]
    c_ttfts = [r.claude_ttft_s for r in paired_results]
    g_totals = [r.groq_total_s for r in paired_results]
    c_totals = [r.claude_total_s for r in paired_results]

    g_ttft_pct = calc_percentiles(g_ttfts)
    c_ttft_pct = calc_percentiles(c_ttfts)
    g_tot_pct = calc_percentiles(g_totals)
    c_tot_pct = calc_percentiles(c_totals)

    # Paired differences: Claude - Groq
    ttft_diffs = [c - g for c, g in zip(c_ttfts, g_ttfts)]
    tot_diffs = [c - g for c, g in zip(c_totals, g_totals)]

    ttft_mean_diff, ttft_ci_low, ttft_ci_high = bootstrap_ci(ttft_diffs)
    tot_mean_diff, tot_ci_low, tot_ci_high = bootstrap_ci(tot_diffs)

    wilcoxon_ttft = stats.wilcoxon(c_ttfts, g_ttfts)
    wilcoxon_tot = stats.wilcoxon(c_totals, g_totals)

    # Accuracy & Groundedness
    g_correctness = sum(1 for r in paired_results if r.groq_is_correct) / n
    c_correctness = sum(1 for r in paired_results if r.claude_is_correct) / n
    g_groundedness = sum(1 for r in paired_results if r.groq_is_grounded) / n
    c_groundedness = sum(1 for r in paired_results if r.claude_is_grounded) / n

    unans = [r for r in paired_results if not r.is_answerable]
    g_ref_acc = sum(1 for r in unans if r.groq_refusal_accurate) / len(unans) if unans else 1.0
    c_ref_acc = sum(1 for r in unans if r.claude_refusal_accurate) / len(unans) if unans else 1.0

    # Token & Cost Projections per 1,000 queries
    avg_g_in = mean([r.groq_in_tokens for r in paired_results])
    avg_g_out = mean([r.groq_out_tokens for r in paired_results])
    avg_c_in = mean([r.claude_in_tokens for r in paired_results])
    avg_c_out = mean([r.claude_out_tokens for r in paired_results])

    g_pricing = PRICING[GROQ_MODEL]
    c_pricing = PRICING[ANTHROPIC_MODEL]

    g_cost_per_1k = (avg_g_in * g_pricing["input_per_m"] + avg_g_out * g_pricing["output_per_m"]) / 1000.0
    c_cost_per_1k = (avg_c_in * c_pricing["input_per_m"] + avg_c_out * c_pricing["output_per_m"]) / 1000.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "models": {"groq": GROQ_MODEL, "claude": ANTHROPIC_MODEL},
        "ttft": {
            "groq_p50_s": g_ttft_pct["p50"],
            "groq_p95_s": g_ttft_pct["p95"],
            "claude_p50_s": c_ttft_pct["p50"],
            "claude_p95_s": c_ttft_pct["p95"],
            "mean_delta_claude_minus_groq_s": ttft_mean_diff,
            "delta_95ci_low_s": ttft_ci_low,
            "delta_95ci_high_s": ttft_ci_high,
            "wilcoxon_p": float(wilcoxon_ttft.pvalue),
        },
        "total_latency": {
            "groq_p50_s": g_tot_pct["p50"],
            "groq_p95_s": g_tot_pct["p95"],
            "claude_p50_s": c_tot_pct["p50"],
            "claude_p95_s": c_tot_pct["p95"],
            "mean_delta_claude_minus_groq_s": tot_mean_diff,
            "delta_95ci_low_s": tot_ci_low,
            "delta_95ci_high_s": tot_ci_high,
            "wilcoxon_p": float(wilcoxon_tot.pvalue),
        },
        "eval_scores": {
            "groq_correctness": g_correctness,
            "claude_correctness": c_correctness,
            "groq_groundedness": g_groundedness,
            "claude_groundedness": c_groundedness,
            "groq_refusal_accuracy": g_ref_acc,
            "claude_refusal_accuracy": c_ref_acc,
        },
        "cost_per_1k_queries": {
            "groq_published_usd": g_cost_per_1k,
            "claude_published_usd": c_cost_per_1k,
            "cost_ratio_claude_to_groq": c_cost_per_1k / g_cost_per_1k if g_cost_per_1k > 0 else 0.0,
        },
        "item_details": [asdict(r) for r in paired_results],
    }

    # Save artifact
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"paired_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = asyncio.run(run_paired_benchmark())

    print("\n" + "=" * 60)
    print("PARLEY STEP 4: PAIRED EVALUATION REPORT")
    print(f"Sample: n={rep['n']} matched items (21 answerable, 5 unanswerable)")
    print(f"Models: Groq='{rep['models']['groq']}' vs Claude='{rep['models']['claude']}'")
    print("=" * 60)
    print("1. LATENCY METRICS:")
    print(f"  Groq TTFT:    p50={rep['ttft']['groq_p50_s']:.3f}s | p95={rep['ttft']['groq_p95_s']:.3f}s")
    print(f"  Claude TTFT:  p50={rep['ttft']['claude_p50_s']:.3f}s | p95={rep['ttft']['claude_p95_s']:.3f}s")
    print(f"  TTFT Delta (Claude - Groq): {rep['ttft']['mean_delta_claude_minus_groq_s']:+.3f}s")
    print(f"  95% CI: [{rep['ttft']['delta_95ci_low_s']:+.3f}s, {rep['ttft']['delta_95ci_high_s']:+.3f}s] (Wilcoxon p={rep['ttft']['wilcoxon_p']:.4e})")
    print(f"  Total Latency Delta: {rep['total_latency']['mean_delta_claude_minus_groq_s']:+.3f}s")
    print(f"  95% CI: [{rep['total_latency']['delta_95ci_low_s']:+.3f}s, {rep['total_latency']['delta_95ci_high_s']:+.3f}s] (Wilcoxon p={rep['total_latency']['wilcoxon_p']:.4e})")
    print("\n2. ACCURACY & GROUNDEDNESS:")
    print(f"  Correctness:       Groq = {rep['eval_scores']['groq_correctness']:.4f} | Claude = {rep['eval_scores']['claude_correctness']:.4f}")
    print(f"  Groundedness:      Groq = {rep['eval_scores']['groq_groundedness']:.4f} | Claude = {rep['eval_scores']['claude_groundedness']:.4f}")
    print(f"  Refusal Accuracy:  Groq = {rep['eval_scores']['groq_refusal_accuracy']:.4f} | Claude = {rep['eval_scores']['claude_refusal_accuracy']:.4f}")
    print("\n3. COST PROJECTION (per 1,000 queries):")
    print(f"  Groq:   ${rep['cost_per_1k_queries']['groq_published_usd']:.4f}")
    print(f"  Claude: ${rep['cost_per_1k_queries']['claude_published_usd']:.4f} ({rep['cost_per_1k_queries']['cost_ratio_claude_to_groq']:.1f}x higher)")
    print("=" * 60 + "\n")

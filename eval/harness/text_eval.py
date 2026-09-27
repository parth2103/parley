"""Minimal text-mode evaluation harness for Parley RAG and LLM groundedness.

Evaluates models outside Pipecat with retrieved context (top-3 chunks):
1. Correctness: matches required facts/numbers and avoids trap values.
2. Groundedness: verifies all cited numbers/currencies exist in retrieved context or query.
3. Refusal accuracy: verifies refusal on unanswerable out-of-scope items without hallucinating coverage.
4. Latency: TTFT and total completion latency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from backend.rag.corpus import CORPUS, PolicyChunk
from backend.rag.retriever import HybridRetriever
from eval.scenarios.golden_set import GOLDEN_SET, GoldenItem

load_dotenv()

SYSTEM_PROMPT = (
    "You are Parley, a voice assistant for an insurance policy & claims line.\n"
    "Answer the customer's question directly using ONLY the policy information provided in the context below.\n"
    "If the answer cannot be determined from the context, clearly state that the policy does not cover it "
    "or that information is not available in the policy documents.\n"
    "Do not assume or extrapolate. Never invent coverage limits, dollar amounts, or deductibles.\n"
    "Keep your response concise and direct (under 3 sentences), suitable to be spoken aloud over the phone."
)

REFUSAL_PATTERN = re.compile(
    r"(not covered|no coverage|does not cover|not included|cannot be determined|"
    r"not available|not mentioned|out of scope|neither .* covers|uncovered|exclude[ds]?|"
    r"separate (?:flood|earthquake|commercial|pet|travel)? ?policy|not provided)",
    re.IGNORECASE,
)


def extract_numbers_and_amounts(text: str) -> set[str]:
    """Extract normalized numbers, percentages, and dollar amounts."""
    # Remove commas from numbers like 25,000 -> 25000, 1,350 -> 1350
    normalized = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Match numbers (integers or decimals)
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", normalized)
    return set(matches)


@dataclass
class ItemEvalResult:
    item_id: str
    category: str
    question: str
    is_answerable: bool
    retrieved_chunk_ids: list[str]
    retrieval_success: bool
    response: str
    ttft_s: float
    total_time_s: float
    is_correct: bool
    is_grounded: bool
    hallucinated_numbers: list[str]
    is_refused: bool
    refusal_accurate: bool
    prompt_tokens: int
    completion_tokens: int


class DeterministicScorer:
    """Zero-dependency, objective rule-based scorer for facts, numbers, and groundedness."""

    @staticmethod
    def score_item(
        item: GoldenItem,
        response: str,
        retrieved_chunks: list[PolicyChunk],
    ) -> tuple[bool, bool, list[str], bool, bool]:
        """Returns: (is_correct, is_grounded, hallucinated_numbers, is_refused, refusal_accurate)."""
        response_nums = extract_numbers_and_amounts(response)
        
        # Grounded universe = numbers in retrieved context + numbers mentioned in customer question
        context_text = " ".join([c.content for c in retrieved_chunks])
        context_nums = extract_numbers_and_amounts(context_text)
        question_nums = extract_numbers_and_amounts(item.question)
        grounded_universe = context_nums | question_nums

        # Groundedness: any number in response that is NOT in context or question is a hallucination
        hallucinated_numbers = sorted(list(response_nums - grounded_universe))
        is_grounded = len(hallucinated_numbers) == 0

        # Refusal detection
        is_refused = bool(REFUSAL_PATTERN.search(response))

        if not item.is_answerable:
            # Unanswerable query: must refuse and not invent dollar amounts / coverage limits
            refusal_accurate = is_refused and is_grounded
            is_correct = refusal_accurate
        else:
            # Answerable query: must have all required numbers and zero forbidden numbers
            refusal_accurate = not is_refused  # should NOT refuse valid answerable items
            
            # Check required numbers
            has_required = True
            for req in item.required_numbers:
                req_norm = req.replace(",", "")
                if req_norm not in response_nums:
                    has_required = False
                    break

            # Check forbidden trap numbers
            has_forbidden = False
            for forb in item.forbidden_numbers:
                forb_norm = forb.replace(",", "")
                if forb_norm in response_nums:
                    has_forbidden = True
                    break

            is_correct = has_required and (not has_forbidden) and (not is_refused)

        return is_correct, is_grounded, hallucinated_numbers, is_refused, refusal_accurate


def format_context_prompt(question: str, chunks: list[PolicyChunk]) -> str:
    context_str = "\n\n".join(
        [f"[{c.chunk_id}] {c.title}\n{c.content}" for c in chunks]
    )
    return (
        f"Context:\n{context_str}\n\n"
        f"Customer Question: {question}\n\n"
        "Answer directly and concisely for a phone caller:"
    )


async def call_groq_streaming(client, model: str, prompt: str) -> tuple[float, float, str, int, int]:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.0,
        stream=True,
    )
    first_token_time = None
    chunks = []
    async for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            chunks.append(chunk.choices[0].delta.content)
    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    full_text = "".join(chunks).strip()
    # Estimate token counts: ~4 chars per token heuristic if not returned in stream
    est_prompt_tokens = len(SYSTEM_PROMPT + prompt) // 4
    est_completion_tokens = len(full_text) // 4
    return ttft, total_time, full_text, est_prompt_tokens, est_completion_tokens


async def call_claude_streaming(client, model: str, prompt: str) -> tuple[float, float, str, int, int]:
    t0 = time.perf_counter()
    stream = await client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0,
        stream=True,
    )
    first_token_time = None
    chunks = []
    async for chunk in stream:
        if chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
            if chunk.delta.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t0
                chunks.append(chunk.delta.text)
    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    full_text = "".join(chunks).strip()
    est_prompt_tokens = len(SYSTEM_PROMPT + prompt) // 4
    est_completion_tokens = len(full_text) // 4
    return ttft, total_time, full_text, est_prompt_tokens, est_completion_tokens


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    if len(values) == 1:
        return {"p50": values[0], "p95": values[0]}
    cuts = quantiles(values, n=100, method="inclusive")
    return {"p50": median(values), "p95": cuts[94]}


async def run_eval(
    model_provider: str,
    top_k: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    retriever = HybridRetriever(CORPUS)
    scorer = DeterministicScorer()
    results: list[ItemEvalResult] = []

    client = None
    model_id = ""

    if not dry_run:
        if model_provider == "groq":
            import openai
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY is missing from environment")
            client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            model_id = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"
        elif model_provider == "claude":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is missing from environment")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            model_id = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"
        else:
            raise ValueError(f"Unknown model provider: {model_provider}")

    for idx, item in enumerate(GOLDEN_SET, start=1):
        retrieved_pairs = retriever.retrieve(item.question, top_k=top_k)
        retrieved = [chunk for chunk, _score in retrieved_pairs]
        retrieved_ids = [c.chunk_id for c in retrieved]

        # Check retrieval success
        if item.is_answerable:
            retrieval_success = any(t in retrieved_ids for t in item.target_chunk_ids)
        else:
            retrieval_success = True

        prompt = format_context_prompt(item.question, retrieved)

        if dry_run:
            # Deterministic mock responses for testing harness correctness
            if not item.is_answerable:
                response = "I am sorry, but that service is not covered under your policy."
            else:
                # generate response containing required facts
                response = f"Your policy states coverage with required terms: {' '.join(item.required_numbers)}."
            ttft_s = 0.05
            total_time_s = 0.10
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response) // 4
        else:
            if model_provider == "groq":
                ttft_s, total_time_s, response, prompt_tokens, completion_tokens = await call_groq_streaming(
                    client, model_id, prompt
                )
            else:
                ttft_s, total_time_s, response, prompt_tokens, completion_tokens = await call_claude_streaming(
                    client, model_id, prompt
                )

        is_corr, is_grnd, hall_nums, is_ref, ref_acc = scorer.score_item(item, response, retrieved)

        results.append(
            ItemEvalResult(
                item_id=item.id,
                category=item.category,
                question=item.question,
                is_answerable=item.is_answerable,
                retrieved_chunk_ids=retrieved_ids,
                retrieval_success=retrieval_success,
                response=response,
                ttft_s=ttft_s,
                total_time_s=total_time_s,
                is_correct=is_corr,
                is_grounded=is_grnd,
                hallucinated_numbers=hall_nums,
                is_refused=is_ref,
                refusal_accurate=ref_acc,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    # Compute summary metrics
    n = len(results)
    retrieval_acc = sum(1 for r in results if r.retrieval_success) / n
    correctness = sum(1 for r in results if r.is_correct) / n
    groundedness = sum(1 for r in results if r.is_grounded) / n
    hallucination_rate = 1.0 - groundedness

    unanswerable = [r for r in results if not r.is_answerable]
    refusal_acc = sum(1 for r in unanswerable if r.refusal_accurate) / len(unanswerable) if unanswerable else 1.0

    ttfts = [r.ttft_s for r in results]
    totals = [r.total_time_s for r in results]
    ttft_pct = calculate_percentiles(ttfts)
    total_pct = calculate_percentiles(totals)

    return {
        "n": n,
        "model_provider": model_provider,
        "model_id": model_id,
        "retrieval_recall_top_k": retrieval_acc,
        "correctness": correctness,
        "groundedness": groundedness,
        "hallucination_rate": hallucination_rate,
        "refusal_accuracy": refusal_acc,
        "ttft_p50_s": ttft_pct["p50"],
        "ttft_p95_s": ttft_pct["p95"],
        "total_latency_p50_s": total_pct["p50"],
        "total_latency_p95_s": total_pct["p95"],
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parley Minimal Text-Mode Eval Harness")
    parser.add_argument("--model", choices=["groq", "claude"], default="groq")
    parser.add_argument("--dry-run", action="store_true", help="Run harness with mock responses without API spend")
    args = parser.parse_args()

    summary = asyncio.run(run_eval(model_provider=args.model, dry_run=args.dry_run))

    print(f"\n==========================================")
    print(f"PARLEY TEXT EVALUATION SUMMARY (n={summary['n']})")
    print(f"Provider: {summary['model_provider']} ({summary['model_id'] or 'dry-run'})")
    print(f"==========================================")
    print(f"Retrieval Recall@3:   {summary['retrieval_recall_top_k']:.4f}")
    print(f"Correctness:          {summary['correctness']:.4f}")
    print(f"Groundedness:         {summary['groundedness']:.4f}")
    print(f"Hallucination Rate:   {summary['hallucination_rate']:.4f}")
    print(f"Refusal Accuracy:     {summary['refusal_accuracy']:.4f}")
    print(f"TTFT (p50 / p95):     {summary['ttft_p50_s']:.3f}s / {summary['ttft_p95_s']:.3f}s")
    print(f"Total (p50 / p95):    {summary['total_latency_p50_s']:.3f}s / {summary['total_latency_p95_s']:.3f}s")
    print(f"==========================================\n")

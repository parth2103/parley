"""Bare-API TTFT Diagnostic: Groq vs Claude outside Pipecat.

Interleaved, streaming, warm client, matched 20 turns on fictional insurance queries.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from statistics import mean, median, quantiles
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You are a test voice agent. Keep replies under 2 sentences."
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"

PROMPTS = [
    "What is the standard deductible for comprehensive auto coverage on policy POL-4401?",
    "How do I report a chipped windshield claim?",
    "Is water backup damage covered under my standard homeowner policy?",
    "What documents do I need to submit after a minor fender bender?",
    "Can I add rental car reimbursement to my active auto policy?",
    "How long do I have to file a claim after property theft?",
    "Does my liability coverage protect me if I borrow a friend's vehicle?",
    "What is the difference between collision and comprehensive coverage?",
    "Will my monthly premium increase if I file a glass repair claim?",
    "How can I obtain an official proof of insurance card for the DMV?",
    "Does my policy cover roadside assistance and towing?",
    "What steps should I take immediately after a home pipe freeze burst?",
    "Can I pay my annual insurance premium in quarterly installments?",
    "What is an umbrella policy, and when does it apply?",
    "How does an adjuster determine if my vehicle is a total loss?",
    "Is damage from an earthquake included in standard property insurance?",
    "How do I update the primary garaging address on my vehicle?",
    "Are stolen personal belongings inside my parked car covered by auto or home insurance?",
    "What happens if the other driver involved in an accident is uninsured?",
    "How can I check the real-time status of my open claim CLM-8821?",
]


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot calculate percentiles without values")
    if len(values) == 1:
        return {"p50": values[0], "p95": values[0]}
    cuts = quantiles(values, n=100, method="inclusive")
    return {"p50": median(values), "p95": cuts[94]}


async def query_claude(client, prompt: str) -> tuple[float, float, str]:
    t0 = time.perf_counter()
    stream = await client.messages.create(
        model=ANTHROPIC_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        stream=True,
    )
    first_token_time = None
    text_chunks = []
    async for chunk in stream:
        if chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
            if chunk.delta.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t0
                text_chunks.append(chunk.delta.text)
    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    return ttft, total_time, "".join(text_chunks)


async def query_groq(client, prompt: str) -> tuple[float, float, str]:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        stream=True,
    )
    first_token_time = None
    text_chunks = []
    async for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            text_chunks.append(chunk.choices[0].delta.content)
    total_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else total_time
    return ttft, total_time, "".join(text_chunks)


async def run_diagnostic():
    import anthropic
    import openai

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not anthropic_key or not groq_key:
        print("Missing ANTHROPIC_API_KEY or GROQ_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    claude_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
    groq_client = openai.AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    print(f"Audited Models: Claude='{ANTHROPIC_MODEL}', Groq='{GROQ_MODEL}'")
    print(f"Number of test items: {len(PROMPTS)}")
    print("Performing client warmup...")
    await query_claude(claude_client, "Warmup query.")
    await query_groq(groq_client, "Warmup query.")
    print("Warmup complete. Starting interleaved test runs...\n")

    groq_ttfts: list[float] = []
    claude_ttfts: list[float] = []
    paired_diffs: list[float] = []

    for i, prompt in enumerate(PROMPTS, start=1):
        # Interleave execution: alternate which provider goes first on each turn
        if i % 2 == 1:
            g_ttft, g_total, g_text = await query_groq(groq_client, prompt)
            c_ttft, c_total, c_text = await query_claude(claude_client, prompt)
        else:
            c_ttft, c_total, c_text = await query_claude(claude_client, prompt)
            g_ttft, g_total, g_text = await query_groq(groq_client, prompt)

        diff = c_ttft - g_ttft
        groq_ttfts.append(g_ttft)
        claude_ttfts.append(c_ttft)
        paired_diffs.append(diff)

        print(
            f"Turn {i:02d} | Groq TTFT: {g_ttft:.4f}s (tot: {g_total:.3f}s) | "
            f"Claude TTFT: {c_ttft:.4f}s (tot: {c_total:.3f}s) | Diff: +{diff:.4f}s"
        )
        # Brief pause between turns to mimic natural pacing and avoid burst rate limits
        await asyncio.sleep(0.3)

    g_p = _percentiles(groq_ttfts)
    c_p = _percentiles(claude_ttfts)
    d_p = _percentiles(paired_diffs)

    print("\n" + "=" * 60)
    print("BARE-API TTFT DIAGNOSTIC RESULTS (n=20 matched turns)")
    print("=" * 60)
    print(f"Groq ({GROQ_MODEL}):")
    print(f"  TTFT p50 : {g_p['p50']:.6f} s")
    print(f"  TTFT p95 : {g_p['p95']:.6f} s")
    print(f"  TTFT mean: {mean(groq_ttfts):.6f} s")
    print(f"  TTFT min : {min(groq_ttfts):.6f} s")
    print(f"  TTFT max : {max(groq_ttfts):.6f} s")

    print(f"\nClaude ({ANTHROPIC_MODEL}):")
    print(f"  TTFT p50 : {c_p['p50']:.6f} s")
    print(f"  TTFT p95 : {c_p['p95']:.6f} s")
    print(f"  TTFT mean: {mean(claude_ttfts):.6f} s")
    print(f"  TTFT min : {min(claude_ttfts):.6f} s")
    print(f"  TTFT max : {max(claude_ttfts):.6f} s")

    print("\nPaired Difference (Claude TTFT - Groq TTFT):")
    print(f"  Diff p50 : {d_p['p50']:.6f} s")
    print(f"  Diff p95 : {d_p['p95']:.6f} s")
    print(f"  Diff mean: {mean(paired_diffs):.6f} s")


if __name__ == "__main__":
    asyncio.run(run_diagnostic())

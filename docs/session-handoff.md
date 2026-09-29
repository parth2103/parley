<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-29 16:15, by: Antigravity

## Last completed
- Phase 1 Closeout Steps 1–9 Fully Executed:
  - Step 1: Applied documentation corrections to `PROJECT_CONTEXT.md` and `docs/latency-budget.md`.
  - Step 2: Ran paired discriminative benchmark on full 35-item golden set at $k=3$ ($N=105$ queries/model). Measured Task Correctness (Groq=0.7356 vs Claude=0.8966, paired delta=+0.1609, 95% Bootstrap CI: [+0.0575, +0.2874]), Refusal Accuracy (Groq=0.5000 vs Claude=0.9444), Groundedness (Groq=0.9714 vs Claude=0.9429), Claim Check (0.9885 for both via Qwen 3.8-27B), and TTFT streaming latency (Groq p50=0.480s vs Claude p50=0.973s, mean delta=+1.142s slower for Claude, $p < 10^{-17}$).
  - Step 3: Categorized Groq failures across all 32 misses: 6 retrieval misses (lexical gap on GS-29/31), 16 wrong facts from right chunk (contract figure omissions), 10 refusal phrasing errors.
  - Step 4: Analyzed voice tail ($n=18$ baseline log): isolated the 3.63s p95 to Smart Turn v3 inter-clause silence evaluation (1.9–3.2s pause holds), not STT, LLM, or TTS.
  - Step 5: Investigated tool tail latency on TC-10 (10.97s): isolated to dual-LLM roundtrip cloud API queueing (Step 1: 4.64s, tool exec: 0.02ms, Step 2: 6.33s). Designed 4-tier conversational filler and timeout mitigation.
  - Step 6: Harder tool eval across 21 test cases: Tool Selection Accuracy = 0.9048 (19/21), Argument Extraction Accuracy = 0.8462 (11/13 strict, 12/13 with phone normalization), Clarification Accuracy = 0.7500 (6/8). Tool-call turn latency: p50 = 6.286 s, p95 = 14.408 s.
  - Step 7: Calculated exact cost arithmetic from $n=105$ measured queries: Groq = $0.000135/turn ($0.135/1k queries) vs Claude = $0.002287/turn ($2.287/1k queries), a 16.91x ratio. Verified official rate cards (2026-09-28).
  - Step 8: Configured GitHub Actions CI gate (`.github/workflows/ci.yml`) running 58 offline unit tests + 10-item discriminative smoke eval (`eval/harness/ci_smoke_eval.py`). Gate passed: unit tests 58/58 passing, smoke task correctness = 0.7500 (threshold >= 0.6856), groundedness = 1.0000 (threshold >= 0.9000), CI run cost = $0.00135.
  - Step 9: Re-baselined live voice pipeline with RAG and tools confirmed active in execution path ($n=18$ complete turns): STT p50=0.304s (p95=0.370s), LLM p50=0.709s (p95=15.641s), TTS TTFA p50=0.287s (p95=0.379s), True V2V p50=3.881s (p95=18.108s, mean=5.404s). This forms the official C0 baseline for the Phase 1 latency budget table.

## In progress
- None. Phase 1 closeout complete, verified, and committed.

## Next action (exactly one)
- Begin Phase 2 planning (token/latency governor, semantic caching, multi-agent orchestration) once user approves start.

## Blockers
- None. Phase 1 closeout is complete.

## Pending confirmations (need user OK before writing to logs)
- None. All pending confirmations approved and written to docs/decisions-log.md, docs/work-log.md, docs/latency-budget.md, and PROJECT_CONTEXT.md.

## Contradictions noticed
- None. All sources reconciled.
<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-28 00:50, by: Antigravity

## Last completed
- Closeout Task 1: Defined and measured True Voice-to-Voice ($t_{\text{first\_agent\_audio\_frame\_to\_transport}} - t_{\text{user\_speech\_end\_vad\_stop}}$). Validated with waveform acoustic analysis on 3 turns. Re-reported Groq $n=18$ baseline (p50=1.155s, p95=3.627s vs Claude 2.611s / 7.128s).
- Closeout Task 2: Made quality eval discriminative. Expanded golden set to 30 items. Separated Task Correctness from Refusal Accuracy. Added 3rd-family claim-level check (`qwen/qwen3.8-27b`). Evaluated $k=3$ samples per item ($N=90$). Task Correctness: Groq=0.8133 vs Claude=0.9333; Refusal: Groq=0.6667 vs Claude=0.9333; TTFT delta=-554ms.
- Closeout Task 3: Retrieval. Renamed retriever to "BM25 + hashed n-gram" dense vector retriever via RRF across codebase and test harnesses (with backward-compatible alias `BM25HashedNgramRetriever = HybridRetriever`). Grew golden set to 35 items (29 answerable) with vocabulary gaps and conversational speech phrasing. Measured Recall@1 = 0.7931 (23/29), Recall@3 = 0.9310 (27/29), Recall@5 = 0.9655 (28/29). Proved recall is discriminative and not at ceiling (<1.0) due to lexical mismatch on conversational terms (GS-29, GS-31). 51 unit tests passing.
- Closeout Task 4: Tools. Created standalone JSON-RPC 2.0 stdio MCP server (`backend/tools/mcp_server.py`) exposing the 3 tools (`policy_lookup`, `open_claim`, `schedule_callback`). Wired LLM tool calling with pre-attached async handlers into `LLMContext` in `backend/pipeline/voice.py`. Created evaluation harness (`eval/harness/eval_tools.py`) testing 11 cases (10 tool-calling cases across all 3 tools + 1 missing-info clarification case where policy number is absent). Verified 100% Tool Selection Accuracy (11/11), 100% Argument Extraction Accuracy (10/10), and 100% Clarification Accuracy (1/1, correctly prompting user for policy number without calling a tool). Replaced in-process microsecond metric with end-to-end tool-call turn latency: p50 = 837.1 ms (0.837 s), p90 = 10,966.8 ms, p95 = 10,966.8 ms, mean = 2,621.5 ms. All 58 unit tests passing.

## In progress
- Phase 1 Closeout Complete: Ready for Wrap Protocol to propose log diffs for decisions-log, latency-budget, and PROJECT_CONTEXT.md.
- Files: `PROJECT_CONTEXT.md`, `docs/decisions-log.md`, `docs/latency-budget.md`, `docs/work-log.md`.
- Last command run: `uv run --frozen pytest` (58 passed in 2.97s).
- State: All 4 Phase 1 Closeout tasks completed and verified.

## Next action (exactly one)
- Wrap Protocol: Propose diffs to `PROJECT_CONTEXT.md`, `docs/latency-budget.md`, `docs/decisions-log.md`, and `docs/work-log.md` with measured numbers and wait for user confirmation.

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- Closeout Task 1 True V2V numbers (Groq p50=1.155s, p95=3.627s vs Claude 2.611s / 7.128s) for docs/latency-budget.md.
- Closeout Task 2 discriminative eval results (Groq Task Correctness = 0.8133 vs Claude 0.9333, Refusal = 0.6667 vs 0.9333, TTFT delta = -554ms) for decisions-log and work-log.
- Closeout Task 3 retrieval rename ("BM25 + hashed n-gram") and grown golden set recall (Recall@1=0.7931, Recall@3=0.9310, Recall@5=0.9655) for decisions-log and PROJECT_CONTEXT.md.
- Closeout Task 4 tool eval results (Selection=1.0000, Argument=1.0000, Clarify=1.0000, Tool-Call Turn Latency p50=0.837s, mean=2.622s) for decisions-log and latency-budget.

## Contradictions noticed
- None.
<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-28 00:45, by: Antigravity

## Last completed
- Closeout Task 1: Defined and measured True Voice-to-Voice ($t_{\text{first\_agent\_audio\_frame\_to\_transport}} - t_{\text{user\_speech\_end\_vad\_stop}}$). Validated with waveform acoustic analysis on 3 turns. Re-reported Groq $n=18$ baseline (p50=1.155s, p95=3.627s vs Claude 2.611s / 7.128s).
- Closeout Task 2: Made quality eval discriminative. Expanded golden set to 30 items. Separated Task Correctness from Refusal Accuracy. Added 3rd-family claim-level check (`qwen/qwen3.8-27b`). Evaluated $k=3$ samples per item ($N=90$). Task Correctness: Groq=0.8133 vs Claude=0.9333; Refusal: Groq=0.6667 vs Claude=0.9333; TTFT delta=-554ms.
- Closeout Task 3: Retrieval. Renamed retriever to "BM25 + hashed n-gram" dense vector retriever via RRF across codebase and test harnesses (with backward-compatible alias `BM25HashedNgramRetriever = HybridRetriever`). Grew golden set to 35 items (29 answerable) with vocabulary gaps and conversational speech phrasing. Measured Recall@1 = 0.7931 (23/29), Recall@3 = 0.9310 (27/29), Recall@5 = 0.9655 (28/29). Proved recall is discriminative and not at ceiling (<1.0) due to lexical mismatch on conversational terms (GS-29, GS-31). 51 unit tests passing.

## In progress
- Closeout Task 4: Tools (MCP server confirmation/integration, wire LLM tool calling into voice loop, eval on ~10 tool calls + 1 missing-info case, measure tool-call turn latency).
- Files: `backend/tools/policy_tools.py`, `backend/pipeline/voice.py`, `eval/harness/test_tools_eval.py`.
- Last command run: `uv run python eval/harness/eval_retrieval.py` (completed exit 0).
- State: Task 3 completed and verified.

## Next action (exactly one)
- Closeout Task 4: Tools (Confirm MCP server interface for policy_lookup, open_claim, schedule_callback, wire LLM tool calling into voice loop, create tool eval suite for ~10 cases + 1 clarification case, measure tool-call turn latency).

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- Closeout Task 1 True V2V numbers (Groq p50=1.155s, p95=3.627s) for docs/latency-budget.md.
- Closeout Task 2 discriminative eval results (Groq Task Correctness = 0.8133 vs Claude 0.9333, Refusal = 0.6667 vs 0.9333, TTFT delta = -554ms) for decisions-log and work-log.
- Closeout Task 3 retrieval rename ("BM25 + hashed n-gram") and grown golden set recall (Recall@1=0.7931, Recall@3=0.9310, Recall@5=0.9655) for decisions-log and PROJECT_CONTEXT.md.

## Contradictions noticed
- None.
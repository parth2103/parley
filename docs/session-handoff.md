<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-28 00:05, by: Antigravity

## Last completed
- Step 0: Model string audit confirming `claude-sonnet-4-6` sent by config and measured in $n=18$ baseline.
- Step 1: Bare-API TTFT diagnostic outside Pipecat ($n=20$ matched turns, Groq p50=0.493s, Claude p50=0.833s).
- Step 2: Fictional policy corpus (4 docs, 12 chunks with numerical traps) and hybrid retrieval (Okapi BM25 + dense vector + RRF). Evaluation verified Recall@3 = 1.0000 (10/10) and Recall@5 = 1.0000 (10/10).
- Step 3: Golden evaluation set (26 items: 21 answerable + 5 unanswerable) and text-mode eval harness with deterministic scoring.
- Step 4: Paired LLM comparison ($n=26$ matched interleaved items, Groq `openai/gpt-oss-120b` vs Claude `claude-sonnet-4-6`). Both models achieved 1.0000 Groundedness (0 hallucinated numbers/dates). Groq delivered -335ms TTFT advantage (p50=0.509s vs 0.815s, 95% CI: [+0.222s, +0.468s], Wilcoxon $p=4.08\times 10^{-6}$) and -1.576s total latency advantage at 17.2x lower cost ($0.1265 vs $2.1797 per 1k queries). Claude showed higher completion robustness (0 truncations vs 1 empty response and 1 truncation on Groq with max_tokens=150).
- Step 5: Fictional insurance mock MCP tools implemented (`backend/tools/policy_tools.py`) for `policy_lookup`, `open_claim`, and `schedule_callback` with full argument validation, MCP tool schemas (`TOOL_SCHEMAS`), execution dispatcher, and sub-millisecond execution latency ($p50 \le 0.004\text{ms}$). 16 unit tests passing (`backend/tests/test_tools.py`), full suite passing (50 passed).

## In progress
- Step: Step 5 completed; awaiting user confirmation and "go" for Step 6 (End-to-end voice re-baseline, 18-turn C0 ablation).
- Files: `backend/tools/policy_tools.py`, `backend/tests/test_tools.py`.
- Last command run: `uv run --frozen pytest` (50 passed in 3.41s).
- State: MCP tools built, tested, and benchmarked. 0 new dependencies added.

## Next action (exactly one)
- Step 6: End-to-end voice re-baseline (18-turn C0 ablation in live pipeline comparing Groq voice baseline vs Claude original baseline).

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- Step 0: Documenting `claude-sonnet-4-6` audit in decisions-log and work-log.
- Step 1: Documenting bare-API TTFT diagnostic (Groq p50=0.493s, Claude p50=0.833s) in decisions-log and work-log.
- Step 2: Documenting hybrid retrieval results (n=10, Recall@3=1.0000, Recall@5=1.0000) in decisions-log and work-log.
- Step 3: Documenting golden set dataset (n=26) and eval harness in decisions-log and work-log.
- Step 4: Documenting paired LLM comparison results (n=26, Groq p50=0.509s vs Claude p50=0.815s, TTFT delta=-335ms [95% CI: +0.222s, +0.468s], Groundedness=1.0000 for both, Groq primary confirmed with max_tokens fix recommendation) in decisions-log and latency-budget.
- Step 5: Documenting mock MCP tools (`policy_lookup`, `open_claim`, `schedule_callback`) with $n=16$ unit tests and execution latency ($p50 \le 0.004\text{ms}$) in decisions-log and work-log.

## Contradictions noticed
- None remaining.
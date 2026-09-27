<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-26 23:40, by: Antigravity

## Last completed
- Step 0: Model string audit confirming `claude-sonnet-4-6` sent by config and measured in $n=18$ baseline.
- Step 1: Bare-API TTFT diagnostic outside Pipecat ($n=20$ matched turns, Groq p50=0.493s, Claude p50=0.833s).
- Step 2: Fictional policy corpus (4 docs, 12 chunks with numerical traps) and hybrid retrieval (Okapi BM25 + dense vector + RRF). Evaluation verified Recall@3 = 1.0000 (10/10) and Recall@5 = 1.0000 (10/10).
- Step 3: Golden evaluation set (26 items: 21 answerable + 5 unanswerable) and text-mode eval harness with deterministic scoring for correctness, groundedness (number/date hallucination detection), refusal accuracy, and streaming latency. Retriever verified Recall@3 = 1.0000 (21/21) on answerable golden items. Unit tests passing (34 passed).

## In progress
- Step: Step 3 completed; awaiting user review of judge proposal & cost estimate before running paired Step 4.
- Files: `eval/scenarios/golden_set.py`, `eval/scenarios/golden_set.json`, `eval/harness/text_eval.py`, `backend/tests/test_golden_set.py`.
- Last command run: `uv run --frozen pytest` (34 passed).
- State: Golden set and eval harness complete, verified via dry run and pytest. 0 new dependencies added.

## Next action (exactly one)
- Step 4: Run paired LLM comparison (Groq `openai/gpt-oss-120b` vs Claude `claude-sonnet-4-6`) on golden set with 95% CIs.

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- Step 0: Documenting `claude-sonnet-4-6` audit in decisions-log and work-log.
- Step 1: Documenting bare-API TTFT diagnostic (Groq p50=0.493s, Claude p50=0.833s) in decisions-log and work-log.
- Step 2: Documenting hybrid retrieval results (n=10, Recall@3=1.0000, Recall@5=1.0000) in decisions-log and work-log.
- Step 3: Documenting golden set dataset (n=26) and eval harness with Recall@3 = 1.0000 (21/21) in decisions-log and work-log.

## Contradictions noticed
- None remaining.
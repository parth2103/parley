<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-28 00:30, by: Antigravity

## Last completed
- Closeout Task 1: Defined and measured True Voice-to-Voice ($t_{\text{first\_agent\_audio\_frame\_to\_transport}} - t_{\text{user\_speech\_end\_vad\_stop}}$) in `backend/pipeline/metrics.py`, `backend/pipeline/turn_lifecycle.py`, and `scripts/parse_baseline.py`. Validated on 3 audio turns with acoustic waveform analysis (`eval/harness/validate_waveform.py`). Re-reported Groq $n=18$ baseline: True V2V p50 = 1.155s, p95 = 3.627s (vs Claude baseline p50 = 2.611s, p95 = 7.128s, a 55.7% p50 reduction and 49.1% p95 tail reduction).

## In progress
- Closeout Task 2: Make quality eval discriminative (expected facts scoring, separate task correctness vs refusal accuracy, NLI/judge check for non-numeric claims, harder items: paraphrases, multi-chunk answers, ASR noise, k=3 samples per item, re-run paired Groq vs Claude benchmark).
- Files: `eval/scenarios/golden_set.py`, `eval/harness/text_eval.py`, `eval/harness/paired_comparison.py`.
- Last command run: `uv run python eval/harness/validate_waveform.py` (completed exit 0).
- State: Task 1 verified and committed.

## Next action (exactly one)
- Closeout Task 2: Update golden set and harness with discriminative scoring (question-specific facts, claim-level check, harder multi-chunk and ASR noisy items, k=3 samples) and re-run paired benchmark.

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- True Voice-to-Voice definition ($t_{\text{first\_audio}} - t_{\text{vad\_stop}}$) and $n=18$ measurements (Groq p50=1.155s, p95=3.627s vs Claude p50=2.611s, p95=7.128s) to be updated in `docs/latency-budget.md` and `docs/decisions-log.md`.
- Waveform validation results (3 turns, acoustic onset discrepancy avg = 131ms due to Cartesia initial silence padding) to be noted in work-log.

## Contradictions noticed
- None.
<!-- docs/session-handoff.md -->
# Session handoff (overwrite each session; history lives in git + work-log)

Last updated: 2026-09-28 00:35, by: Antigravity

## Last completed
- Closeout Task 1: Defined and measured True Voice-to-Voice ($t_{\text{first\_agent\_audio\_frame\_to\_transport}} - t_{\text{user\_speech\_end\_vad\_stop}}$). Validated with waveform acoustic analysis on 3 turns. Re-reported Groq $n=18$ baseline (p50=1.155s, p95=3.627s vs Claude 2.611s / 7.128s).
- Closeout Task 2: Made quality eval discriminative. Expanded golden set to 30 items (including multi-chunk reasoning and noisy ASR). Separated Task Correctness from Refusal Accuracy. Added 3rd-family claim-level check (`qwen/qwen3.8-27b`). Evaluated $k=3$ samples per item ($N=90$ evaluations per model). Achieved discriminative scores: Task Correctness = 0.8133 (Groq) vs 0.9333 (Claude); Refusal Accuracy = 0.6667 (Groq) vs 0.9333 (Claude); Claim Check = 0.9733 (Groq) vs 0.9867 (Claude). Confirmed Groq's +554ms TTFT advantage (p50=0.463s vs 0.977s, 95% CI: [+0.472s, +0.641s]).

## In progress
- Closeout Task 3: Retrieval updates (rename retriever "BM25 + hashed n-gram" across docs; grow retrieval test set with hard negatives/distractors so recall is not at ceiling).
- Files: `backend/rag/retriever.py`, `eval/harness/eval_retrieval.py`, `docs/`.
- Last command run: `uv run python eval/harness/discriminative_paired_eval.py` (completed exit 0).
- State: Task 2 completed and verified.

## Next action (exactly one)
- Closeout Task 3: Retrieval (Rename retriever "BM25 + hashed n-gram" in docs, add distractor chunks / hard negative queries to test set so recall is not at ceiling).

## Blockers
- None.

## Pending confirmations (need user OK before writing to logs)
- Closeout Task 1 True V2V numbers (Groq p50=1.155s, p95=3.627s) for docs/latency-budget.md.
- Closeout Task 2 discriminative eval results (Groq Task Correctness = 0.8133 vs Claude 0.9333, Refusal = 0.6667 vs 0.9333, TTFT delta = -554ms) for decisions-log and work-log.

## Contradictions noticed
- None.
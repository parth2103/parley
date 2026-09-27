# Parley work log

This is the ongoing record of implementation, troubleshooting, and verification.
Update it as work progresses. Completed implementation does not imply a verified
live call; live results are recorded separately.

## Current status

- [x] Build the Day-1 browser STT → LLM → TTS skeleton.
- [x] Verify pipeline construction with provider clients mocked and network access blocked.
- [x] Confirm the browser can establish a WebRTC connection during user testing.
- [x] Fix the certificate configuration that blocked STT and TTS connections.
- [x] Retest a live call after restarting the server with the certificate fix.
- [x] Confirm a spoken prompt produces an audible reply.
- [x] Confirm actual stage metrics appear in the terminal during a successful call.
- [x] Add explicit turn start, completion, and incomplete-reason lifecycle logs.
- [x] Add and unit-test the offline baseline parser.
- [x] Collect the user-run batch of real calls for the first latency baseline.

## 2026-09-21 — Baseline instrumentation

- Added `turn_start`, `turn_complete`, and `turn_incomplete` lifecycle records
  around the existing stage metric lines. Incomplete records include missing
  stages and any captured pipeline-error source without copying provider error text.
- Added `scripts/parse_baseline.py` for raw files, pasted log text, or stdin.
  It excludes dropped turns, calculates inclusive p50/p95 percentiles, and
  calculates total percentiles from per-turn stage sums.
- Added a clearly synthetic parser fixture with complete and incomplete turns.
- No live call was run, no baseline data was supplied, and
  `docs/latency-budget.md` remains unchanged.

## 2026-09-22 — Groq ultra-low-latency provider integration

- Added `LLM_PROVIDER=groq` support in `backend/pipeline/config.py` and `backend/pipeline/voice.py`,
  using `openai/gpt-oss-120b` via Pipecat's `GroqLLMService`.
- Added safety timeout to `disconnect_all_sessions` in `backend/api/main.py` to prevent
  lingering jobs from blocking incoming browser connections.
- Offline tests updated: 25 tests passed (`uv run --frozen pytest`).
- Conducted live spoken test with Groq in `logs/parley_2026-09-22T23-13-03-438421.log`:
  - 4 complete turns, 0 dropped turns (100% completion rate).
  - LLM TTFB dropped from 1.476s (Sonnet) to **0.258s (Groq)** — an **82.5% reduction**.
  - Total voice-to-voice latency dropped from 1.986s (Sonnet) to **0.822s (Groq)** — a **58.6% reduction**.
  - Multi-turn conversation turns completed at 800ms, 844ms, and 799ms, maintaining sub-second
    latency without context-growth delays.

## 2026-09-22 — Baseline collection and metrics verification
 
- Collected real billed baseline across 23 attempted turns ($n=18$ complete)
  in `logs/parley_2026-09-22T21-53-32-890873.log`.
- Resolved session #0 turn-gap question from prior run: In the prior rejected run,
  Turns 2-10 in Session #0 recorded `missing_stages=LLM,STT,TTS` because
  Pipecat's `report_only_initial_ttfb=True` blocked metric callbacks after Turn 1.
  With `report_only_initial_ttfb=False` active in this run, Turns 1, 2, 3, 4, 5, 6,
  9, 10, and 11 all captured complete STT, LLM, and TTS metrics. Turns 7 and 8 were
  incomplete due to user speech starting before the LLM finished
  (`superseded_by_next_turn;missing_stages=LLM,TTS`), confirming that metric
  callback suppression is fully resolved.
- Verified pooled latencies: STT TTFB p50=0.303s (p95=0.382s), LLM TTFB p50=1.476s
  (p95=4.318s), TTS TTFB p50=0.141s (p95=0.208s), TTS TTFA p50=0.281s (p95=0.355s),
  and Total voice-to-voice p50=1.986s (p95=4.929s).
- Identified Claude Sonnet 4.6 as the primary pipeline bottleneck (>75% of latency).
- Documented findings in `docs/latency-budget.md` and `docs/decisions-log.md`.

## 2026-09-22 — Connection recovery

- Fixed a stale-session failure where refreshing or closing the browser could
  leave the server rejecting every later connection as already active.
- A new Connect attempt now closes an existing server-side test call before
  starting its replacement, preserving the one-call-at-a-time limit.
- Added an offline unit test for orphaned-session cleanup; no provider clients
  are constructed and no billed calls are made by this test.

## 2026-09-17 — Initial implementation

- Set up Python 3.12, uv, the dependency lockfile, and Pipecat pinned to 1.7.0.
- Built FastAPI signaling and a static browser page using SmallWebRTC.
- Configured Silero VAD with `stop_secs=0.2` and local Smart Turn v3 ONNX.
- Connected Deepgram Nova-3, Anthropic `claude-sonnet-4-6` with thinking disabled,
  and Cartesia Sonic 3.5 in the pipeline.
- Added an explicit ElevenLabs Flash v2.5 setup fallback.
- Added startup configuration checks, secret-file exclusions, and billing reminders.
- Enabled Pipecat stage metrics without publishing latency estimates or results.
- Added Docker configuration, setup documentation, and the decisions log.
- Kept agents, tools, retrieval, and evaluation directories empty except for `.gitkeep`.
- Added an explicit free NLTK tokenizer-data installation step and startup check.

## 2026-09-17 — Troubleshooting during testing

| Issue | Finding | Action / status |
| --- | --- | --- |
| Browser reported “site can’t be reached” | No server was listening, and `.env` was missing. | Provided local configuration and server startup steps. Subsequent browser connection confirmed progress. |
| Server refused startup | `CARTESIA_VOICE_ID` was missing or still a placeholder. | Explained how to select a Cartesia voice and configure its ID. Subsequent startup succeeded. |
| No spoken reply after connecting | Browser connected, but server logs showed certificate verification failures for both STT and TTS. | Added certifi as a direct dependency and configured its trusted CA bundle at startup. Explicit trust-store settings are preserved; verification remains enabled. Live retest pending. |
| Audio control appeared paused | Playback state could be obscured by the connection-status message. | Added a separate visible paused-audio notice. This was a UI improvement, not the cause established by the provider-error logs. |

Also explained Anthropic token-based pricing. The pricing example was
illustrative, not a measured project cost.

## Verification record

- **2026-09-17:** Latest offline suite: **9 tests passed**, using
  `uv run --frozen pytest -q backend/tests/test_pipeline_imports.py`.
  Coverage includes both TTS pipeline configurations, missing credentials,
  turn-count deduplication, trusted certificate loading, and preservation of
  explicit trust stores. Provider clients are mocked; tests make no API calls.
- **2026-09-17:** Checked browser JavaScript syntax with `node --check`,
  Compose configuration with `docker compose config --no-env-resolution`,
  and whitespace with `git diff --check`.
- **2026-09-17:** Verified application startup and shutdown with dummy
  configuration without starting a provider call.
- **2026-09-22:** Orphaned-session cleanup test passed (`1 passed`); Python
  compilation and `git diff --check` also passed.
- **2026-09-22:** Full offline test suite passed with 25 tests (`uv run --frozen pytest`),
  covering Groq pipeline construction, settings validation, and custom model overrides.
- **2026-09-22:** Live audible cloud round trip and Day-1 voice latency baseline verified
  across 18 complete turns ($n=18$, Claude Sonnet 4.6). Total voice-to-voice p50=1.986s, p95=4.929s.
- **2026-09-22:** Live comparative voice testing with Groq (`openai/gpt-oss-120b`) verified
  across 4 complete turns (0 dropped). LLM TTFB reduced by 82.5% to 0.258s; total voice-to-voice
  latency reduced by 58.6% to 0.822s.
- Docker execution has not been verified.

## Next test

1. Continue comparative testing or collect larger-N Groq batch if desired.
2. Maintain Phase 4 retrieval and multi-agent scope paused until approved.

## Deferred scope

RAG, multi-agent orchestration, MCP tools, caching beyond Pipecat defaults,
governor, dashboard, evals, phone number, and local-vs-cloud comparison remain
out of scope for this milestone.

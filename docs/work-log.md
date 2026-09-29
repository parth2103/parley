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
- [x] Execute paired discriminative evaluation on 35 golden cases (N=105) for Groq vs Claude.
- [x] Build and evaluate stdio MCP tools and hybrid RAG in voice pipeline.
- [x] Run harder tool eval across 21 test cases and establish tool latency baseline.
- [x] Set up GitHub Actions CI gate with offline unit tests and discriminative smoke eval.
- [x] Capture C0 live voice re-baseline with RAG and tools active across 18 complete turns.

## 2026-09-29 — Phase 1 closeout and C0 live re-baseline

- Completed all Phase 1 closeout steps:
  - Discriminative eval ($n=105$): Groq 0.7356 vs Claude 0.8966 correctness (+16.1% paired delta, 95% CI: [+0.0575, +0.2874]); Refusal 0.5000 vs 0.9444; Groq streaming TTFT p50=0.480s vs Claude p50=0.973s (+1.142s mean delta).
  - Failure categorization: 32 Groq misses attributed to 6 retrieval misses, 16 wrong facts from right chunk, 10 refusal phrasing errors.
  - Voice tail analysis: Isolated 3.63s p95 tail in initial baseline to Smart Turn v3 silence pause hold (1.9–3.2s).
  - Tool tail analysis: Isolated TC-10 10.97s latency to dual-LLM roundtrip cloud queueing.
  - Harder tool eval ($n=21$): Selection 0.9048 (19/21), Extraction 0.8462 (11/13 strict, 12/13 normalized), Clarification 0.7500 (6/8), Turn latency p50=6.286s, p95=14.408s.
  - Cost arithmetic: Groq $0.1353/1k queries vs Claude $2.2869/1k queries (16.91x ratio uncached).
  - CI gate: GitHub Actions (`.github/workflows/ci.yml`) passing 58 unit tests and 10-case smoke eval (correctness 0.7500 >= 0.6856, cost $0.00135).
  - C0 live re-baseline ($n=18$): Captured live voice turns with RAG and tools confirmed active: STT p50=0.304s (p95=0.370s), LLM p50=0.709s (p95=15.641s), TTS TTFA p50=0.287s (p95=0.379s), True V2V p50=3.881s, p95=18.108s.

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
- **2026-09-28:** Paired discriminative benchmark completed on full 35-item golden set ($n=105$, Groq vs Claude).
- **2026-09-28:** Harder tool evaluation completed across 21 test cases (Selection 0.9048, Extraction 0.8462, Clarification 0.7500).
- **2026-09-29:** Offline unit test suite expanded to 58 tests (`uv run --frozen pytest`), 58/58 passing with zero API calls.
- **2026-09-29:** CI smoke gate executed in GitHub Actions (`eval/harness/ci_smoke_eval.py`), 10 cases passing correctness (0.7500) and groundedness (1.0000).
- **2026-09-29:** Live C0 voice re-baseline captured across 18 complete turns with RAG and tools active in execution path. True V2V p50=3.881s, p95=18.108s.
- Docker execution has not been verified.

## Next test

1. Begin Phase 2 planning (token/latency governor, semantic caching, multi-agent orchestration).
2. Maintain Phase 2 execution paused until approval.

## Deferred scope

LangGraph multi-agent orchestration, token/latency governor, semantic
caching, Next.js dashboard, phone number (Twilio SIP), Spanish support,
returning-caller memory, load testing.

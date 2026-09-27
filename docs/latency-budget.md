# Latency budget

Initial small-N baseline recorded on 2026-09-22 from real billed calls (`logs/parley_2026-09-22T21-53-32-890873.log`).

## Measured baseline (2026-09-22)

Method: `pipecat_builtin` stage metrics observer.
Sample size: n=18 complete turns (5 incomplete turns excluded).
Pipeline: Deepgram Nova-3 STT -> Anthropic Claude Sonnet 4.6 (thinking disabled) -> Cartesia Sonic 3.5 TTS.

| Stage | Target budget (Day 1) | Actual p50 (2026-09-22) | Actual p95 (2026-09-22) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **STT TTFB** | < 0.300 s | 0.3034881353378296 s | 0.3819462895393372 s | Deepgram Nova-3; relative to Silero VAD speech stop |
| **LLM TTFB** | < 0.800 s | 1.4757475852966309 s | 4.317530107498169 s | Anthropic claude-sonnet-4-6; primary pipeline bottleneck |
| **TTS TTFB** | < 0.150 s | 0.14074552059173584 s | 0.2078911542892456 s | Cartesia Sonic 3.5 synthesis request to first byte |
| **TTS TTFA** | < 0.300 s | 0.28114564895629884 s | 0.3548289094924927 s | Cartesia Sonic 3.5 synthesis request to first audio chunk |
| **Total voice-to-voice** | < 1.400 s | 1.9858288993835451 s | 4.928868246936799 s | Summed per-turn stage times (STT + LLM + TTS TTFA) |

*Note: Small-N sample (n=18), not production-representative. Latency is dominated (>75%) by Claude Sonnet 4.6 TTFB.*

## Per-group breakdown

### Group 1: Repeated short phrase (Session #0, n=9 complete)
- STT TTFB: p50 = 0.2619602680206299 s, p95 = 0.38294644355773927 s
- LLM TTFB: p50 = 1.2741320133209229 s, p95 = 2.171447229385376 s
- TTS TTFB: p50 = 0.14099597930908203 s, p95 = 0.21646542549133302 s
- TTS TTFA: p50 = 0.2828710994720459 s, p95 = 0.35705463829040524 s
- Total voice-to-voice: p50 = 1.835807382583618 s, p95 = 2.8065073673248286 s

### Group 2: Varied phrasing (Session #1, n=5 complete)
- STT TTFB: p50 = 0.32009387016296387 s, p95 = 0.3698596000671387 s
- LLM TTFB: p50 = 1.3413879871368408 s, p95 = 3.5913352966308594 s
- TTS TTFB: p50 = 0.14208674430847168 s, p95 = 0.16366376876831054 s
- TTS TTFA: p50 = 0.3022943458557129 s, p95 = 0.3085706203460694 s
- Total voice-to-voice: p50 = 1.9070308933258056 s, p95 = 4.209993599700928 s

### Group 3: Conversation attempts (Sessions #2-#5, n=4 complete)
- STT TTFB: p50 = 0.2854830026626587 s, p95 = 0.3277173161506653 s
- LLM TTFB: p50 = 1.9762474298477173 s, p95 = 4.9897479057312015 s
- TTS TTFB: p50 = 0.13615083694458008 s, p95 = 0.14105534553527832 s
- TTS TTFA: p50 = 0.26828229904174805 s, p95 = 0.287347553062439 s
- Total voice-to-voice: p50 = 2.548660577774048 s, p95 = 5.548427933692932 s

## Methodology and stage definitions

- **STT TTFB**: Pipecat's finalized-transcription timing relative to VAD speech end.
- **LLM TTFB**: Model request to first streamed response chunk.
- **TTS TTFB**: Synthesis request to first byte received from Cartesia.
- **TTS TTFA**: Synthesis request to first audible synthesized audio sample.
- **Total voice-to-voice**: Summed per-turn stages (STT TTFB + LLM TTFB + TTS TTFA), percentile calculated across turns (not sum of stage percentiles).
- Incomplete turns (dropped turns) are excluded from percentile calculation and reported separately with error reasons.

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
- **Total voice-to-voice (Legacy summed method)**: Summed per-turn stages (STT TTFB + LLM TTFB + TTS TTFA), percentile calculated across turns (not sum of stage percentiles). *Superseded by True Voice-to-Voice.*
- Incomplete turns (dropped turns) are excluded from percentile calculation and reported separately with error reasons.

## True Voice-to-Voice Baselines (Current Standard)

Defined as timestamp of user speech end (Silero VAD stop) to timestamp of first agent audio frame dispatched to WebRTC transport ($t_{\text{first\_agent\_audio\_frame\_to\_transport}} - t_{\text{user\_speech\_end\_vad\_stop}}$). Captures utterance-boundary evaluation, transport queuing, and inter-stage serialization.

*Note: The earlier summed-stage figures (e.g. Groq p50 = 0.847 s / p95 = 0.979 s; Claude p50 = 1.986 s / p95 = 4.929 s) are superseded by this measurement.*

| Metric | Groq (`openai/gpt-oss-120b`) | Claude Sonnet 4.6 (`claude-sonnet-4-6`) | Delta (Groq vs Claude) [unpaired, directional] |
| :--- | :--- | :--- | :--- |
| **Sample Size** | n=18 complete turns | n=18 complete turns | Matched sample count |
| **True V2V p50** | 1.155 s | 2.611 s | -1.456 s (-55.7%) |
| **True V2V p90** | 3.329 s | 6.452 s | -3.123 s (-48.4%) |
| **True V2V p95** | 3.627 s | 7.128 s | -3.501 s (-49.1%) |
| **Mean** | 1.547 s | 3.097 s | -1.550 s (-50.1%) |

- **Waveform acoustic validation (n=3)**: Software True V2V = 0.819 s – 1.014 s; acoustic waveform gap = 0.950 s – 1.135 s (discrepancy average = 131.15 ms due to Cartesia leading silence in synthesized audio frames).

## Official Phase 1 C0 Baseline (RAG + Tools in Path, 2026-09-29)

Captured across $n=18$ complete live voice turns (`logs/parley_2026-09-29T15-58-29-356068.log`, artifact: `eval/results/c0_rebaseline_20260929.json`) with Groq `openai/gpt-oss-120b`, Deepgram Nova-3, Cartesia Sonic-3.5, and both RAG (`HybridRetriever`) and tools (`policy_lookup`, `open_claim`, `schedule_callback`) active in the pipeline.

| Config | STT p50 / p95 | LLM p50 / p95 | TTS p50 / p95 | True V2V p50 / p95 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C0 (Phase 1 Baseline)** | **0.304 s / 0.370 s** | **0.709 s / 15.641 s** | **0.287 s / 0.379 s** | **3.881 s / 18.108 s** | Groq gpt-oss-120b, Nova-3, Sonic-3.5, RAG + tools in path (n=18 complete turns, mean True V2V = 5.404 s) |

*Observation: Bimodal distribution. Non-tool conversational queries execute at True V2V $\le 1.38\text{ s}$ (min = 0.984 s). Tail latency is driven by dual-LLM cloud queueing during tool calling (p95 = 15.641 s) and Smart Turn v3 silence evaluation holds during inter-clause pauses.*

## Tool-Call Turn Latency (Harder Evaluation Suite, n=21)

Measures full turn duration from user prompt through initial tool call, in-process execution, and follow-up spoken response generation across 21 test cases (including ambiguous intent, malformed IDs, multi-step requests, missing parameters, future dates, prompt injections, and invalid perils).

- **Sample Size**: n=21 test cases (13 tool-calling turns, 8 clarification/refusal cases)
- **Primary LLM**: Groq `openai/gpt-oss-120b` (Temperature: 0.0)
- **Tool Selection Accuracy**: 0.9048 (19/21)
- **Argument Extraction Accuracy**: 0.8462 (11/13 strict, 12/13 with phone number normalization)
- **Missing-Info Clarification Accuracy**: 0.7500 (6/8)
- **Tool Turn Latency (n=13 tool turns)**:
  - **p50**: 6,285.6 ms (6.286 s)
  - **p90**: 12,197.0 ms (12.197 s)
  - **p95**: 14,408.0 ms (14.408 s)
  - **Mean**: 6,825.7 ms (6.826 s)
  - *In-process execution latency remains deterministic and negligible: p50 = 0.03 ms, p95 = 0.36 ms.*

## Cost Arithmetic (Measured n=105, 2026-09-28)

Calculated from exact prompt and completion token counts on the paired discriminative benchmark ($N=105$ queries per model). Rates from official provider pricing (accessed 2026-09-28: Groq $0.15/$0.60 per 1M; Anthropic $3.00/$15.00 per 1M).

| Metric | Groq (`openai/gpt-oss-120b`) | Claude Sonnet 4.6 (`claude-sonnet-4-6`) | Ratio (Claude / Groq) |
| :--- | :--- | :--- | :--- |
| **Mean Prompt Tokens** | 414.63 | 404.20 | 0.97x |
| **Mean Completion Tokens** | 121.77 | 71.62 | 0.59x |
| **Cost per Turn** | **$0.00013526** (0.0135¢) | **$0.00228690** (0.2287¢) | **16.91x** |
| **Cost per 1,000 Queries** | **$0.1353** | **$2.2869** | **16.91x** |
| **With Prompt Caching (est.)** | $0.1353 / 1k | ~$1.341 / 1k | ~9.9x |

*Voice Infrastructure Context: High-quality neural TTS (Cartesia Sonic at $0.075/1k chars, ~$0.020/turn) dominates unit economics over LLM costs, making the LLM choice a <10% delta in total session cost.*


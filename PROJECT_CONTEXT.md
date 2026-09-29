# Parley — Project Context (read this first, every session)

This file is the single source of truth for any agent starting work on
this repo. Read it before writing code. Update it — don't let it go stale.

## What this is

A latency-budgeted, multi-agent voice agent for an insurance policy &
claims phone line. Portfolio project for an AI Engineer / Full-Stack /
Voice AI job search. Core story: "production agents that hit latency,
cost, and accuracy targets — proven with evals." Every feature is judged
against that story, not against "would this be cool to build."

## Current phase

**Phase 1 (Weeks 1–2): core voice loop + RAG + MCP tools + basic evals.**

Status:
- [x] Voice loop built (Pipecat, SmallWebRTC, Deepgram Nova-3, Cartesia TTS)
- [x] First real latency baseline captured and parsed (n=18, Claude Sonnet 4.6 (claude-sonnet-4-6))
- [x] LLM TTFB root cause isolated: cloud streaming TTFT delta (+1.142s for Claude) vs pause-hold tail in turn detection.
- [x] Groq provider comparison: paired discriminative benchmark completed on full 35-item golden set at k=3 (N=105 queries/model). Task correctness: Groq=0.7356 vs Claude=0.8966 (+16.1% paired delta, 95% Bootstrap CI: [+0.0575, +0.2874]); Refusal accuracy: Groq=0.5000 vs Claude=0.9444.
- [x] RAG: BM25 + hashed n-gram dense vectors via RRF over mock policy docs. Discriminative Recall@1=0.7931, Recall@3=0.9310, Recall@5=0.9655; active in voice pipeline via policy_lookup.
- [x] MCP tool mocks: stdio JSON-RPC 2.0 MCP server exposing policy_lookup, open_claim, schedule_callback; wired to Pipecat FunctionSchema. Harder eval (n=21): Selection 0.9048, Extraction 0.8462, Clarification 0.7500.
- [x] Eval harness & CI gate: GitHub Actions CI (.github/workflows/ci.yml) with 58 offline unit tests + 10-item discriminative smoke gate ($0.00135/run).

Do not start Phase 2 (governor, caching, multi-agent orchestration/
LangGraph) or Phase 3 (dashboard, phone number) without explicit
approval, regardless of how tempting a detour looks.

## Hard rules for any agent working on this repo

1. Never invent latency, cost, or accuracy numbers. Real measurement or a
   clearly labeled `[not yet measured]` placeholder — nothing in between.
2. Never add a new provider, service, dependency, or feature that wasn't
   explicitly asked for in the current task prompt. If it seems like a
   good idea, say so as a suggestion and wait for approval — don't build
   it and present it as done. (This happened once already: an unplanned
   Groq integration got built and framed as a finished comparison. Don't
   repeat that pattern.)
3. Mock/fictional data only — no real insurer data, no real customer
   data, no PII, ever.
4. Before writing to docs/decisions-log.md, docs/work-log.md, or
   docs/latency-budget.md, report findings in chat first and wait for
   confirmation — unless the task prompt explicitly says to write
   directly.
5. Keep engineering-log tone in every doc: no marketing language, no
   superlatives ("breakthrough," "elite tier"), no exclamation points.
   State what was measured, the method, and the sample size. Nothing else.
6. Any comparison between two configs (models, providers, stacks) needs
   matched sample sizes and the same test scenarios/content. If it
   doesn't have that, label the result directional-only, don't present
   it as a conclusion.
7. This repo is fully independent of Quantum Webb's code, data, and
   product. Flag anything that starts to overlap.
8. Default to free tiers and low-cost options. Flag any step that
   incurs real spend, with an estimate.

## Stack (current, pinned — update this section when it changes)

- Framework: Pipecat 1.7.0
- Transport: SmallWebRTC
- VAD: Silero, `stop_secs=0.2`
- Turn detection: Smart Turn v3 (local ONNX)
- STT: Deepgram Nova-3
- LLM: **Groq (`openai/gpt-oss-120b`) is primary for latency/cost; correctness 0.7356 vs Claude 0.8966 (n=105).** Claude Sonnet 4.6 (`claude-sonnet-4-6`) kept behind the same `llm_provider` config as a fallback/comparison path. See Open Questions.
- TTS: Cartesia Sonic 3.5 (fallback: ElevenLabs Flash v2.5)
- Retriever: BM25 + hashed n-gram vectors via RRF (pure Python/NumPy, zero external embedding dependency), integrated in voice tool execution path.
- Tools: exposed via stdio MCP server; voice pipeline calls the same functions directly via Pipecat FunctionSchema

## Open questions / unresolved — check before doing related work

- **Primary LLM quality vs latency trade-off:** Groq delivers sub-second TTFT (p50=0.480s) and $0.135/1k queries, but trails Claude Sonnet 4.6 by 16.1% in task correctness (0.7356 vs 0.8966) and 44.4% in refusal accuracy (0.5000 vs 0.9444). Decision to maintain Groq primary is locked for latency, with Claude available via `LLM_PROVIDER=anthropic`.
- **Live C0 Baseline established:** Groq n=18 live voice turns under True V2V definition with RAG + tools in path: STT p50=0.304s, LLM p50=0.709s, TTS p50=0.287s, True V2V p50=3.881s, p95=18.108s. Tail driven by dual-LLM tool queueing and inter-clause silence evaluation.
- **Session disconnect behavior** (`Media stream error... clearing
  track`) — partially investigated, not conclusively resolved across all
  connection patterns (single-turn vs. multi-turn sessions).

## Where things live

- `docs/decisions-log.md` — dated decisions, one line each:
  `date | decision | rationale | evidence | revisit when`
- `docs/work-log.md` — ongoing implementation/troubleshooting/verification
  record
- `docs/latency-budget.md` — target vs. measured-actual latency per stage
- `logs/` — raw test run logs (gitignored, not committed)
- `scripts/parse_baseline.py` — turns a raw log into a percentile table

## Deferred — do not build without an explicit go-ahead

LangGraph multi-agent orchestration, token/latency governor, semantic
caching, Next.js dashboard, phone number (Twilio SIP), Spanish support,
returning-caller memory, load testing.

## Keeping this file current

Update "Current phase," "Open questions," and "Deferred" whenever a
phase gate is crossed or a question gets resolved. Full history stays in
`decisions-log.md` and `work-log.md` — this file is the fast-read summary
an agent (or you) should check before starting anything new.

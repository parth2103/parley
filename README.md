# Parley

See the [ongoing work log](docs/work-log.md) for completed work, troubleshooting,
verification results, and the next steps.

Parley explores how to make a multi-agent voice assistant responsive within a
measured latency budget. The first milestone is a bare browser-to-voice round
trip: establish that speech recognition, a short model reply, and synthesized
speech work together before adding orchestration. No latency has been measured
and no performance target is claimed yet.

**This project makes real, billed API calls (Anthropic has no free tier). Set a spend cap in your provider consoles before testing.**

## Architecture

```text
Browser microphone / speaker
          |          ^
          | WebRTC   | synthesized audio
          v          |
SmallWebRTC transport (inside FastAPI; no media server)
          |
Deepgram Nova-3 STT
          |
User context + Silero VAD (stop_secs=0.2)
             + Smart Turn v3 (bundled local ONNX, CPU)
          |
Anthropic claude-sonnet-4-6 (thinking disabled)
          |
Cartesia Sonic 3.5 TTS
          |  [manual setup fallback: ElevenLabs Flash v2.5]
          v
SmallWebRTC output -> assistant context

Browser -- POST /api/offer --> FastAPI -- SDP answer --> Browser
```

The model ID is exactly `claude-sonnet-4-6`; this repository uses that ID rather
than the task's conflicting “Sonnet 5” label. The system prompt is:
“You are a test voice agent. Keep replies under 2 sentences.”

Python is restricted to the 3.12 series. Pipecat is pinned to `1.7.0`; `uv.lock`
locks the dependency graph. Its `webrtc` extra supplies SmallWebRTC. Smart Turn's
bundled v3-family CPU ONNX model requires no provider account or inference fee.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
   `uv python install 3.12` and `uv sync --frozen` from this directory.
   Install Pipecat's sentence tokenizer data once (a free public download):
   `uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab`.
   Docker installs this data during its build. Startup checks that it exists.
2. Create accounts at [Deepgram](https://console.deepgram.com/),
   [Anthropic](https://console.anthropic.com/), and
   [Cartesia](https://play.cartesia.ai/). Enable API billing and access to the
   selected models. Create API keys and select a Cartesia voice ID.
3. **Set a spend cap in each provider console before testing.** Check whether
   each account offers a hard cap or only budget alerts. Where no hard cap is
   available, use limited prepaid credit and disable automatic top-ups where
   supported. An alert alone does not stop billing.
4. Run `cp .env.example .env`, then fill the keys and `CARTESIA_VOICE_ID` in
   `.env`. Keep `TTS_PROVIDER=cartesia`. The file is gitignored and excluded
   from the Docker build. Startup rejects absent, empty, or example-placeholder
   credentials before accepting calls. This validates configuration presence;
   provider authentication and model access are checked when connecting.
5. Run the offline smoke test (no API calls; sockets blocked):

   ```sh
   uv run --frozen pytest -q backend/tests/test_pipeline_imports.py
   ```

6. Start the server, ready to make **real, billed calls** when you connect:

   ```sh
   uv run --frozen uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
   ```

   Console output is also captured automatically in a new
   `logs/parley_<timestamp>.log` file for each server start; the active path is
   printed during startup.

7. Open `http://localhost:8000`, click **Connect**, allow microphone access, and
   say a short sentence. The agent waits for your speech; it sends no greeting
   request. Use headphones. Click **Disconnect** immediately after testing:
   Deepgram receives audio while connected, including silence. Watch the
   terminal for provider errors and stage metrics. Only one test call can be
   active at a time. This is a local test server without authentication.

If Cartesia setup fails, create an [ElevenLabs](https://elevenlabs.io/) account,
set its spend controls, and fill `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`
in `.env`. Set `TTS_PROVIDER=elevenlabs` and restart. This selects
`eleven_flash_v2_5`; the unused Cartesia key is then not required. Fallback is
an explicit setup choice, not an automatic mid-call retry against another bill.

### Exact dependency commands

These are the commands to recreate the declared dependency selections; a
checkout should use `uv sync --frozen` instead to retain the lockfile:

```sh
uv add "pipecat-ai[anthropic,cartesia,deepgram,elevenlabs,local-smart-turn,silero,webrtc]==1.7.0"
uv add "certifi>=2026.7.22" "fastapi>=0.141.1" "uvicorn[standard]>=0.53.0" "python-dotenv>=1.2.3"
uv add --dev "pytest>=9.1.1" "pytest-asyncio>=1.4.0" "pytest-socket>=0.8.1"
```

Python installations without configured certificate roots use certifi's trusted
CA bundle at server startup. TLS certificate and hostname verification remain
enabled. Explicit `SSL_CERT_FILE` or `SSL_CERT_DIR` settings are preserved.

### Docker

`docker compose up --build` starts only the FastAPI service and reads `.env`.
WebRTC media uses ephemeral UDP ports in addition to HTTP signaling, so Compose
uses host networking. On Linux this is native; on Docker Desktop enable host
networking under Settings → Resources → Network. If host networking is
unavailable, use the native server command above. The container binds to all
host interfaces; use it only on a trusted development host. This setup has no
STUN/TURN relay and is intended for a browser on the same machine, not remote
NAT traversal. Microphone access requires localhost or HTTPS.

## Console metrics

Pipecat's built-in metrics are enabled. The console formatter prints the first
STT TTFB and LLM TTFB per user turn, plus one TTS line containing both TTFB and
TTFA. Each line carries a date, session ID, turn, stage, model, and
`method=pipecat_builtin`. Initial placeholder metrics are disabled. Sentence
chunks and repeated forwarding of the same metric do not produce extra stage
lines. A stage interrupted before its measurement completes has no result;
missing results are never filled with a made-up value.

TTFA measures first audible synthesized audio inside the service, not audio
heard at the browser. No end-to-end measurement, benchmark, dashboard, or
numeric latency budget exists yet. See [measurement boundaries](docs/latency-budget.md).

## Not yet built

- RAG/retrieval
- Multi-agent orchestration (including LangGraph)
- Governor
- Dashboard
- Evals
- Phone number / telephony
- Local-vs-cloud comparison

There are no MCP tools or application caches. The `agents`, `tools`, `rag`,
and evaluation leaf directories contain only `.gitkeep` files.

## Files

```text
parley/
├── backend/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── voice.py
│   ├── agents/.gitkeep
│   ├── tools/.gitkeep
│   ├── rag/.gitkeep
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── static/index.html
│   └── tests/test_pipeline_imports.py
├── eval/
│   ├── scenarios/.gitkeep
│   └── harness/.gitkeep
├── docs/
│   ├── decisions-log.md
│   └── latency-budget.md
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

Configuration constants, package/model versions, dates, and protocol identifiers
are not latency claims. Any future published latency result must come from a
measured run with its date and method recorded.

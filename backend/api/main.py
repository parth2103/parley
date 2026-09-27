"""Local test API and SmallWebRTC signaling."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from backend.pipeline.config import BILLING_REMINDER, Settings, configure_tls

LOG_FORMAT = "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {message}"


def configure_logging() -> Path:
    """Send each server run to the console and its own local log file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f")
    log_path = logs_dir / f"parley_{timestamp}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.add(file_handler, level="INFO", format=LOG_FORMAT)
    logger.info("Logging to {}", log_path)
    return log_path


async def disconnect_all_sessions(app: FastAPI) -> None:
    """Stop every server-side call, including one orphaned by a page reload."""
    sessions = list(app.state.sessions.items())
    for pc_id, (connection, worker, job) in sessions:
        # Remove it first so the worker's own cleanup can safely do the same.
        app.state.sessions.pop(pc_id, None)
        await worker.cancel()
        await connection.disconnect()
        try:
            await asyncio.wait_for(asyncio.shield(job), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.log_path = configure_logging()
    logger.warning(BILLING_REMINDER)
    load_dotenv()
    configure_tls()
    app.state.settings = Settings.from_env()
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        raise RuntimeError(
            "Parley startup refused: missing NLTK tokenizer data. Run: "
            "uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab"
        ) from None
    # Validate configuration before importing the audio stack or accepting offers.
    from backend.pipeline.voice import build_pipeline
    app.state.build_pipeline = build_pipeline
    app.state.sessions = {}
    app.state.offer_lock = asyncio.Lock()
    logger.info("Parley ready; TTS provider={}. Connect and speak to begin billed calls.", app.state.settings.tts_provider)
    try:
        yield
    finally:
        await disconnect_all_sessions(app)


app = FastAPI(lifespan=lifespan)


class Offer(BaseModel):
    sdp: str
    type: Literal["offer"]


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/offer")
async def offer(body: Offer):
    from pipecat.workers.runner import WorkerRunner
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

    async with app.state.offer_lock:
        if app.state.sessions:
            logger.warning(
                "A new connection replaced {} existing test call(s) left active on the server.",
                len(app.state.sessions),
            )
            await disconnect_all_sessions(app)
        connection = SmallWebRTCConnection(ice_servers=[])
        try:
            await connection.initialize(sdp=body.sdp, type=body.type)
            transport = SmallWebRTCTransport(connection, TransportParams(audio_in_enabled=True, audio_out_enabled=True))
            worker = app.state.build_pipeline(transport, app.state.settings, connection.pc_id)

            @transport.event_handler("on_client_disconnected")
            async def disconnected(_transport, _client):
                await worker.cancel()

            async def run():
                try:
                    await WorkerRunner(handle_sigint=False).run(worker)
                except Exception:
                    logger.error("Voice pipeline failed; check provider configuration and account access.")
                finally:
                    await connection.disconnect()
                    app.state.sessions.pop(connection.pc_id, None)

            answer = connection.get_answer()
            if answer is None:
                raise RuntimeError("SmallWebRTC produced no answer")
            job = asyncio.create_task(run())
            app.state.sessions[connection.pc_id] = (connection, worker, job)
            return answer
        except Exception:
            await connection.disconnect()
            raise


@app.delete("/api/sessions/{pc_id}", status_code=204)
async def disconnect(pc_id: str):
    session = app.state.sessions.get(pc_id)
    if session:
        connection, worker, job = session
        await worker.cancel()
        await connection.disconnect()
        await asyncio.gather(job, return_exceptions=True)
    return Response(status_code=204)

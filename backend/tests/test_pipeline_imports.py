"""Offline construction checks: real processors, mocked cloud clients, blocked sockets."""

from unittest.mock import MagicMock

import pytest

from backend.pipeline.config import Settings


@pytest.mark.parametrize("provider,model", [("cartesia", "sonic-3.5"), ("elevenlabs", "eleven_flash_v2_5")])
async def test_pipeline_builds_without_api_calls(monkeypatch, provider, model):
    # Pipecat tries to download tokenizer data at import. Construction does not
    # tokenize speech, so suppress that optional resource download in this test.
    monkeypatch.setattr("nltk.download", lambda *args, **kwargs: False)
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor
    from backend.pipeline.voice import build_pipeline

    deepgram = MagicMock()
    anthropic = MagicMock()
    cartesia_connect = MagicMock(side_effect=AssertionError("Must not connect to Cartesia"))
    elevenlabs_connect = MagicMock(side_effect=AssertionError("Must not connect to ElevenLabs"))
    monkeypatch.setattr("pipecat.services.deepgram.stt.AsyncDeepgramClient", deepgram)
    monkeypatch.setattr("pipecat.services.anthropic.llm.AsyncAnthropic", anthropic)
    monkeypatch.setattr("pipecat.services.cartesia.tts.CartesiaTTSService._websocket_connect", cartesia_connect)
    monkeypatch.setattr("pipecat.services.elevenlabs.tts.ElevenLabsTTSService._websocket_connect", elevenlabs_connect)
    transport = MagicMock()
    transport.input.return_value = FrameProcessor(name="test-input")
    transport.output.return_value = FrameProcessor(name="test-output")
    worker = build_pipeline(transport, Settings("test", "test", "test", "test-voice", provider), "offline")
    assert isinstance(worker, PipelineWorker)
    processors = worker.pipeline.processors_with_metrics()
    services = {p.name: p for p in processors if p.name in {"STT", "LLM", "TTS"}}
    assert services["STT"]._settings.model == "nova-3"
    assert services["LLM"]._settings.model == "claude-sonnet-4-6"
    assert services["LLM"]._settings.thinking.type == "disabled"
    assert services["TTS"]._settings.model == model
    assert worker._params.enable_metrics
    assert not worker._params.report_only_initial_ttfb
    assert not worker._params.send_initial_empty_metrics
    deepgram.assert_called_once()
    anthropic.assert_called_once()
    assert not deepgram.return_value.mock_calls
    anthropic.return_value.beta.messages.create.assert_not_called()
    cartesia_connect.assert_not_called()
    elevenlabs_connect.assert_not_called()
    await worker.pipeline.cleanup()
    await worker.cleanup()


@pytest.mark.asyncio
async def test_custom_anthropic_model_configured_and_honored(monkeypatch):
    monkeypatch.setattr("nltk.download", lambda *args, **kwargs: False)
    from pipecat.processors.frame_processor import FrameProcessor
    from backend.pipeline.voice import build_pipeline

    deepgram = MagicMock()
    anthropic = MagicMock()
    cartesia_connect = MagicMock()
    monkeypatch.setattr("pipecat.services.deepgram.stt.AsyncDeepgramClient", deepgram)
    monkeypatch.setattr("pipecat.services.anthropic.llm.AsyncAnthropic", anthropic)
    monkeypatch.setattr("pipecat.services.cartesia.tts.CartesiaTTSService._websocket_connect", cartesia_connect)
    transport = MagicMock()
    transport.input.return_value = FrameProcessor(name="test-input")
    transport.output.return_value = FrameProcessor(name="test-output")
    settings = Settings("test", "test", "test", "test-voice", "cartesia", anthropic_model="claude-sonnet-4-6")
    worker = build_pipeline(transport, settings, "offline")
    processors = worker.pipeline.processors_with_metrics()
    services = {p.name: p for p in processors if p.name in {"STT", "LLM", "TTS"}}
    assert services["LLM"]._settings.model == "claude-sonnet-4-6"
    await worker.pipeline.cleanup()
    await worker.cleanup()


def test_missing_keys_fail_before_a_call(monkeypatch):
    for name in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    with pytest.raises(RuntimeError, match="Parley startup refused") as error:
        Settings.from_env()
    for name in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "CARTESIA_API_KEY"):
        assert name in str(error.value)


def test_fallback_requires_its_own_credentials(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        Settings.from_env()


def test_anthropic_model_from_env(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "test-voice")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    s1 = Settings.from_env()
    assert s1.anthropic_model == "claude-sonnet-4-6"

    monkeypatch.setenv("ANTHROPIC_MODEL", "custom-model")
    s2 = Settings.from_env()
    assert s2.anthropic_model == "custom-model"


@pytest.mark.asyncio
async def test_groq_pipeline_builds_without_api_calls(monkeypatch):
    monkeypatch.setattr("nltk.download", lambda *args, **kwargs: False)
    from pipecat.processors.frame_processor import FrameProcessor
    from backend.pipeline.voice import build_pipeline
    from pipecat.services.groq.llm import GroqLLMService

    deepgram = MagicMock()
    cartesia_connect = MagicMock()
    monkeypatch.setattr("pipecat.services.deepgram.stt.AsyncDeepgramClient", deepgram)
    monkeypatch.setattr("pipecat.services.cartesia.tts.CartesiaTTSService._websocket_connect", cartesia_connect)
    transport = MagicMock()
    transport.input.return_value = FrameProcessor(name="test-input")
    transport.output.return_value = FrameProcessor(name="test-output")
    settings = Settings(
        "test-deepgram", "", "test-cartesia", "test-voice", "cartesia",
        llm_provider="groq", groq_api_key="gsk_test", groq_model="llama-3.3-70b-versatile",
    )
    worker = build_pipeline(transport, settings, "offline")
    processors = worker.pipeline.processors_with_metrics()
    services = {p.name: p for p in processors if p.name in {"STT", "LLM", "TTS"}}
    assert isinstance(services["LLM"], GroqLLMService)
    assert services["LLM"]._settings.model == "llama-3.3-70b-versatile"
    await worker.pipeline.cleanup()
    await worker.cleanup()


def test_groq_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "test-voice")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    s = Settings.from_env()
    assert s.llm_provider == "groq"
    assert s.groq_api_key == "gsk_test_key"
    assert s.groq_model == "openai/gpt-oss-120b"


def test_groq_missing_key_fails(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "test-voice")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        Settings.from_env()


async def test_startup_rejects_missing_configuration(monkeypatch):
    from backend.api.main import app, lifespan

    monkeypatch.setattr("backend.api.main.load_dotenv", lambda: None)
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        async with lifespan(app):
            pytest.fail("Server accepted missing credentials")


async def test_turn_broadcast_is_counted_once():
    from pipecat.frames.frames import UserStartedSpeakingFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from backend.pipeline.metrics import StageMetricsObserver

    observer = StageMetricsObserver("offline")
    source = FrameProcessor()
    destination = FrameProcessor()
    downstream = UserStartedSpeakingFrame()
    upstream = UserStartedSpeakingFrame()
    for frame, direction in [
        (downstream, FrameDirection.DOWNSTREAM),
        (upstream, FrameDirection.UPSTREAM),
        (downstream, FrameDirection.DOWNSTREAM),
        (upstream, FrameDirection.UPSTREAM),
    ]:
        await observer.on_push_frame(FramePushed(source, destination, frame, direction, 0))
    assert observer.turn == 1


def test_turn_lifecycle_logs_completion(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from pipecat.frames.frames import MetricsFrame, UserStartedSpeakingFrame
    from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection
    from backend.pipeline.metrics import StageMetricsObserver

    lines = []
    monkeypatch.setattr(
        "backend.pipeline.metrics.logger.info",
        lambda template, *args: lines.append(template.format(*args)),
    )
    observer = StageMetricsObserver("lifecycle")
    destination = SimpleNamespace(name="destination")

    async def push(source_name, frame):
        source = SimpleNamespace(name=source_name)
        await observer.on_push_frame(
            FramePushed(source, destination, frame, FrameDirection.DOWNSTREAM, 0)
        )

    async def exercise():
        await push("input", UserStartedSpeakingFrame())
        await push("STT", MetricsFrame(data=[TTFBMetricsData(processor="STT", value=0.1)]))
        await push("LLM", MetricsFrame(data=[TTFBMetricsData(processor="LLM", value=0.2)]))
        await push(
            "TTS",
            MetricsFrame(
                data=[
                    TTFAMetricsData(
                        processor="TTS", ttfb=0.2, ttfa=0.3, leading_silence=0.1
                    )
                ]
            ),
        )

    asyncio.run(exercise())

    assert lines[0] == "session=lifecycle turn=1 event=turn_start"
    assert lines[-1] == "session=lifecycle turn=1 event=turn_complete"
    assert sum("stage=" in line for line in lines) == 3


def test_turn_lifecycle_logs_incomplete_reason(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from pipecat.frames.frames import ErrorFrame, UserStartedSpeakingFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection
    from backend.pipeline.metrics import StageMetricsObserver

    lines = []
    monkeypatch.setattr(
        "backend.pipeline.metrics.logger.info",
        lambda template, *args: lines.append(template.format(*args)),
    )
    observer = StageMetricsObserver("lifecycle")
    source = SimpleNamespace(name="LLM")
    destination = SimpleNamespace(name="destination")

    async def push(frame):
        await observer.on_push_frame(
            FramePushed(source, destination, frame, FrameDirection.DOWNSTREAM, 0)
        )

    async def exercise():
        await push(UserStartedSpeakingFrame())
        await push(ErrorFrame(error="provider details are not copied", processor=source))
        await push(UserStartedSpeakingFrame())

    asyncio.run(exercise())

    incomplete = next(line for line in lines if "event=turn_incomplete" in line)
    assert "reason=superseded_by_next_turn" in incomplete
    assert "missing_stages=LLM,STT,TTS" in incomplete
    assert "last_error=nonfatal_error:LLM" in incomplete
    assert "provider details" not in incomplete


def test_turn_lifecycle_rejects_late_out_of_order_metrics():
    from backend.pipeline.turn_lifecycle import TurnLifecycle

    lifecycle = TurnLifecycle("offline", lambda _line: None)
    lifecycle.start()

    assert not lifecycle.expects_stage("LLM")
    assert lifecycle.expects_stage("STT")
    lifecycle.record_stage("STT")
    assert lifecycle.expects_stage("LLM")


def test_tls_uses_trusted_roots_with_verification_enabled(monkeypatch):
    import os
    import ssl
    import certifi
    from backend.pipeline.config import configure_tls

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    configure_tls()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    context = ssl.create_default_context()
    assert context.get_ca_certs()
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_tls_preserves_explicit_trust_store(monkeypatch, variable):
    import os
    from backend.pipeline.config import configure_tls

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv(variable, "/custom/trust-store")
    configure_tls()
    assert os.environ[variable] == "/custom/trust-store"
    if variable == "SSL_CERT_DIR":
        assert "SSL_CERT_FILE" not in os.environ

"""Day-one STT -> LLM -> TTS pipeline. Construction makes no API calls."""

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from backend.pipeline.config import Settings
from backend.pipeline.metrics import StageMetricsObserver
from backend.tools.policy_tools import get_policy_function_schemas

SYSTEM_PROMPT = (
    "You are Parley, an AI voice assistant for insurance policy & claims phone lines. "
    "Keep replies concise and under 2 sentences suitable for natural speech. "
    "Use search_policy_documents to retrieve coverage rules, exclusions, and guidelines, "
    "policy_lookup to inspect coverage/deductibles for specific policy numbers, "
    "open_claim to initiate FNOL claim filings, "
    "and schedule_callback to book adjuster callbacks. "
    "If required details (such as policy number) are missing, ask the user to clarify before calling a tool."
)


def build_pipeline(transport: BaseTransport, settings: Settings, session: str) -> PipelineWorker:
    stt = DeepgramSTTService(
        name="STT", api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(model="nova-3"),
    )
    if settings.llm_provider == "groq":
        from pipecat.services.groq.llm import GroqLLMService
        llm = GroqLLMService(
            name="LLM", api_key=settings.groq_api_key,
            settings=GroqLLMService.Settings(
                model=settings.groq_model, system_instruction=SYSTEM_PROMPT,
            ),
        )
    else:
        llm = AnthropicLLMService(
            name="LLM", api_key=settings.anthropic_api_key,
            settings=AnthropicLLMService.Settings(
                model=settings.anthropic_model, system_instruction=SYSTEM_PROMPT,
                thinking=AnthropicLLMService.ThinkingConfig(type="disabled"),
            ),
        )
    if settings.tts_provider == "cartesia":
        tts = CartesiaTTSService(
            name="TTS", api_key=settings.tts_api_key,
            settings=CartesiaTTSService.Settings(model="sonic-3.5", voice=settings.tts_voice_id),
        )
    else:
        tts = ElevenLabsTTSService(
            name="TTS", api_key=settings.tts_api_key,
            settings=ElevenLabsTTSService.Settings(model="eleven_flash_v2_5", voice=settings.tts_voice_id),
        )
    tools = get_policy_function_schemas()
    context = LLMContext(tools=tools)
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )
    pipeline = Pipeline([
        transport.input(), stt, aggregators.user(), llm, tts,
        transport.output(), aggregators.assistant(),
    ])
    return PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, report_only_initial_ttfb=False, send_initial_empty_metrics=False),
        observers=[StageMetricsObserver(session)],
        enable_rtvi=False,
        enable_turn_tracking=False,
    )

"""Print Pipecat measurements and the lifecycle around each measured turn."""

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    MetricsFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

from backend.pipeline.turn_lifecycle import TurnLifecycle


class StageMetricsObserver(BaseObserver):
    def __init__(self, session: str):
        super().__init__()
        self.session = session
        self._last_start = None
        self._lifecycle = TurnLifecycle(session, logger.info)

    @property
    def turn(self) -> int:
        return self._lifecycle.turn

    def _close_incomplete_turn(self, reason: str) -> None:
        self._lifecycle.close_incomplete(reason)

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if (
            isinstance(frame, UserStartedSpeakingFrame)
            and data.direction == FrameDirection.DOWNSTREAM
            and frame.id != self._last_start
        ):
            self._last_start = frame.id
            self._lifecycle.start()
            return
        if isinstance(frame, ErrorFrame) and not self._lifecycle.closed:
            processor = frame.processor.name if frame.processor else data.source.name
            self._lifecycle.record_error(processor, frame.fatal)
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            terminal = "pipeline_cancelled" if isinstance(frame, CancelFrame) else "pipeline_ended"
            if frame.reason:
                terminal += "_with_reason"
            self._close_incomplete_turn(terminal)
            return
        if not isinstance(frame, MetricsFrame):
            return
        for metric in frame.data:
            stage = metric.processor
            # Only the originating push; metrics traverse many processors.
            if data.source.name != stage or not self._lifecycle.expects_stage(stage):
                continue
            if stage in {"STT", "LLM"} and isinstance(metric, TTFBMetricsData):
                values = f"ttfb_s={metric.value}"
            elif stage == "TTS" and isinstance(metric, TTFAMetricsData):
                values = f"ttfb_s={metric.ttfb} ttfa_s={metric.ttfa}"
            else:
                continue
            logger.info(
                "session={} turn={} stage={} model={} method=pipecat_builtin {}",
                self.session, self.turn, stage, metric.model, values,
            )
            self._lifecycle.record_stage(stage)

"""Provider-independent state for one measured voice turn at a time."""

from collections.abc import Callable

STAGES = frozenset({"STT", "LLM", "TTS"})


class TurnLifecycle:
    def __init__(self, session: str, emit: Callable[[str], None]):
        self.session = session
        self.emit = emit
        self.turn = 0
        self.reported: set[str] = set()
        self.closed = True
        self.last_error: str | None = None

    def start(self) -> None:
        self.close_incomplete("superseded_by_next_turn")
        self.turn += 1
        self.reported.clear()
        self.last_error = None
        self.closed = False
        self.emit(f"session={self.session} turn={self.turn} event=turn_start")

    def record_error(self, processor: str, fatal: bool) -> None:
        severity = "fatal" if fatal else "nonfatal"
        self.last_error = f"{severity}_error:{processor}"
        if fatal:
            self.close_incomplete("fatal_pipeline_error")

    def record_stage(self, stage: str) -> None:
        if self.closed or stage not in STAGES:
            return
        self.reported.add(stage)
        if self.reported == STAGES:
            self.emit(f"session={self.session} turn={self.turn} event=turn_complete")
            self.closed = True

    def expects_stage(self, stage: str) -> bool:
        """Reject a metric from an older response that crosses a turn boundary."""
        order = ("STT", "LLM", "TTS")
        expected = order[len(self.reported)] if len(self.reported) < len(order) else None
        return not self.closed and stage == expected

    def close_incomplete(self, reason: str) -> None:
        if self.turn == 0 or self.closed:
            return
        missing = ",".join(sorted(STAGES - self.reported))
        parts = [reason, f"missing_stages={missing or 'none'}"]
        if self.last_error:
            parts.append(f"last_error={self.last_error}")
        self.emit(
            f"session={self.session} turn={self.turn} "
            f"event=turn_incomplete reason={';'.join(parts)}"
        )
        self.closed = True

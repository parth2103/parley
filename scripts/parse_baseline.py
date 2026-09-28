#!/usr/bin/env python3
"""Parse Parley's raw console logs into latency percentiles and dropped turns."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, quantiles

STAGES = ("STT", "LLM", "TTS")
STAGE_RE = re.compile(
    r"(?:(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)\s+\|\s+INFO\s+\|\s+)?"
    r"session=(?P<session>\S+)\s+turn=(?P<turn>\d+)\s+stage=(?P<stage>STT|LLM|TTS|TRUE_V2V)\b"
    r"(?P<fields>.*?)(?=$|\r?\n)"
)
VALUE_RE = re.compile(r"\b(?P<name>ttfb_s|ttfa_s|true_v2v_s|v2v_s)=(?P<value>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
LIFECYCLE_RE = re.compile(
    r"(?:(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)\s+\|\s+INFO\s+\|\s+)?"
    r"session=(?P<session>\S+)\s+turn=(?P<turn>\d+)\s+event=(?P<event>turn_start|turn_complete|turn_incomplete)"
    r"(?:\s+reason=(?P<reason>\S+))?"
    r"(?:.*?\b(?:true_v2v_s|v2v_s)=(?P<v2v>\d+(?:\.\d+)?))?"
)


@dataclass
class Turn:
    session: str
    number: int
    stages: dict[str, float] = field(default_factory=dict)
    tts_ttfb: float | None = None
    true_v2v: float | None = None
    lifecycle: str | None = None
    reason: str | None = None
    stt_ts: float | None = None
    tts_ts: float | None = None


@dataclass
class Baseline:
    complete: list[Turn]
    dropped: list[Turn]
    percentiles: dict[str, dict[str, float]]


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot calculate percentiles without values")
    if len(values) == 1:
        return {"p50": values[0], "p95": values[0]}
    cuts = quantiles(values, n=100, method="inclusive")
    return {"p50": median(values), "p95": cuts[94]}


def _parse_ts(ts_str: str | None) -> float | None:
    if not ts_str:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return None


def parse_log(text: str) -> Baseline:
    turns: dict[tuple[str, int], Turn] = {}

    for line in text.splitlines():
        stage_match = STAGE_RE.search(line)
        if stage_match:
            key = (stage_match["session"], int(stage_match["turn"]))
            turn = turns.setdefault(key, Turn(*key))
            values = {match["name"]: float(match["value"]) for match in VALUE_RE.finditer(stage_match["fields"])}
            stage = stage_match["stage"]
            ts = _parse_ts(stage_match.group("timestamp"))

            if stage == "TRUE_V2V":
                for k in ("true_v2v_s", "v2v_s"):
                    if k in values:
                        turn.true_v2v = values[k]
                continue

            value_name = "ttfa_s" if stage == "TTS" else "ttfb_s"
            if value_name in values:
                turn.stages.setdefault(stage, values[value_name])
            if stage == "TTS" and "ttfb_s" in values:
                turn.tts_ttfb = values["ttfb_s"]
            if stage == "STT" and ts is not None:
                turn.stt_ts = ts
            if stage == "TTS" and ts is not None:
                turn.tts_ts = ts

        lifecycle_match = LIFECYCLE_RE.search(line)
        if lifecycle_match:
            key = (lifecycle_match["session"], int(lifecycle_match["turn"]))
            turn = turns.setdefault(key, Turn(*key))
            event = lifecycle_match["event"]
            if event in {"turn_complete", "turn_incomplete"}:
                turn.lifecycle = event
                turn.reason = lifecycle_match["reason"]
            if lifecycle_match.group("v2v"):
                turn.true_v2v = float(lifecycle_match.group("v2v"))

    complete: list[Turn] = []
    dropped: list[Turn] = []
    required = set(STAGES)
    for turn in turns.values():
        has_all_stages = set(turn.stages) == required and turn.tts_ttfb is not None
        if has_all_stages and turn.lifecycle != "turn_incomplete":
            # If true_v2v wasn't explicitly logged, calculate from wall-clock timestamps:
            # VAD stop = stt_ts - stt_ttfb; First audio = tts_ts; v2v = tts_ts - (stt_ts - stt_ttfb)
            if turn.true_v2v is None and turn.stt_ts and turn.tts_ts and "STT" in turn.stages:
                turn.true_v2v = turn.tts_ts - (turn.stt_ts - turn.stages["STT"])
            complete.append(turn)
        else:
            if not turn.reason:
                turn.reason = "unknown"
            dropped.append(turn)

    percentiles: dict[str, dict[str, float]] = {}
    if complete:
        for stage in ("STT", "LLM"):
            percentiles[stage] = _percentiles([turn.stages[stage] for turn in complete])
        percentiles["TTS TTFB"] = _percentiles([turn.tts_ttfb for turn in complete])
        percentiles["TTS TTFA"] = _percentiles([turn.stages["TTS"] for turn in complete])
        totals = [sum(turn.stages[stage] for stage in STAGES) for turn in complete]
        percentiles["Total voice-to-voice"] = _percentiles(totals)
        true_v2vs = [turn.true_v2v for turn in complete if turn.true_v2v is not None]
        if true_v2vs:
            percentiles["True voice-to-voice"] = _percentiles(true_v2vs)

    return Baseline(complete=complete, dropped=dropped, percentiles=percentiles)


def render_markdown(baseline: Baseline) -> str:
    lines = [
        "| Stage | Actual p50 | Actual p95 |",
        "| --- | ---: | ---: |",
    ]
    if baseline.complete:
        display_labels = ["STT", "LLM", "TTS TTFB", "TTS TTFA"]
        if "True voice-to-voice" in baseline.percentiles:
            display_labels.append("True voice-to-voice")
        display_labels.append("Total voice-to-voice")

        for label in display_labels:
            values = baseline.percentiles[label]
            lines.append(f"| {label} | {values['p50']:.6g} s | {values['p95']:.6g} s |")
    else:
        lines.append("| No complete turns | — | — |")

    lines.extend(["", f"Complete turns: {len(baseline.complete)}", f"Dropped turns: {len(baseline.dropped)}"])
    for turn in baseline.dropped:
        missing = ", ".join(stage for stage in STAGES if stage not in turn.stages) or "none"
        lines.append(
            f"- session={turn.session} turn={turn.number}: reason={turn.reason or 'unknown'}; missing={missing}"
        )
    return "\n".join(lines)


def read_input(source: str | None) -> str:
    if source is None or source == "-":
        return sys.stdin.read()
    if "\n" in source or "\r" in source:
        return source
    path = Path(source)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        # Long pasted logs are not valid paths on every platform.
        pass
    return source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a Parley raw log file, pasted log argument, or stdin."
    )
    parser.add_argument("source", nargs="?", help="Log file path or pasted log text; omit for stdin")
    args = parser.parse_args()
    baseline = parse_log(read_input(args.source))
    print(render_markdown(baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

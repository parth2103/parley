from pathlib import Path

import pytest

from scripts.parse_baseline import parse_log, read_input, render_markdown


FIXTURE = Path(__file__).parent / "fixtures" / "baseline.log"


def test_parse_baseline_uses_complete_turns_and_per_turn_totals():
    baseline = parse_log(FIXTURE.read_text())

    assert len(baseline.complete) == 2
    assert len(baseline.dropped) == 1
    assert baseline.dropped[0].reason == (
        "pipeline_cancelled;missing_stages=LLM,TTS;last_error=nonfatal_error:LLM"
    )
    assert baseline.percentiles["STT"] == pytest.approx({"p50": 0.15, "p95": 0.195})
    assert baseline.percentiles["LLM"] == pytest.approx({"p50": 0.3, "p95": 0.39})
    assert baseline.percentiles["TTS TTFB"] == pytest.approx({"p50": 0.375, "p95": 0.4875})
    assert baseline.percentiles["TTS TTFA"] == pytest.approx({"p50": 0.45, "p95": 0.585})
    # Per-turn sums are 0.60 and 1.20. Percentiles are calculated from those
    # totals, rather than by adding independently computed stage percentiles.
    assert baseline.percentiles["Total voice-to-voice"] == pytest.approx(
        {"p50": 0.9, "p95": 1.17}
    )


def test_render_includes_actuals_columns_and_drop_reason():
    output = render_markdown(parse_log(read_input(str(FIXTURE))))

    assert "| Stage | Actual p50 | Actual p95 |" in output
    assert "Complete turns: 2" in output
    assert "Dropped turns: 1" in output
    assert "reason=pipeline_cancelled;missing_stages=LLM,TTS" in output


def test_missing_lifecycle_reason_is_unknown():
    baseline = parse_log(
        "session=old-log turn=7 stage=STT model=nova-3 method=pipecat_builtin ttfb_s=0.1"
    )

    assert len(baseline.dropped) == 1
    assert baseline.dropped[0].reason == "unknown"


def test_pasted_multiline_log_text_is_accepted():
    pasted = "session=pasted turn=1 event=turn_start\nsession=pasted turn=1 event=turn_incomplete"

    assert read_input(pasted) == pasted

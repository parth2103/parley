from backend.pipeline.turn_lifecycle import TurnLifecycle


def test_complete_turn_has_one_start_and_completion():
    lines = []
    lifecycle = TurnLifecycle("test-session", lines.append)

    lifecycle.start()
    lifecycle.record_stage("STT")
    lifecycle.record_stage("LLM")
    lifecycle.record_stage("TTS")
    lifecycle.record_stage("TTS")

    assert lines == [
        "session=test-session turn=1 event=turn_start",
        "session=test-session turn=1 event=turn_complete",
    ]


def test_superseded_turn_reports_missing_stages_and_error_source():
    lines = []
    lifecycle = TurnLifecycle("test-session", lines.append)

    lifecycle.start()
    lifecycle.record_stage("STT")
    lifecycle.record_error("LLM", fatal=False)
    lifecycle.start()

    assert lines[1] == (
        "session=test-session turn=1 event=turn_incomplete "
        "reason=superseded_by_next_turn;missing_stages=LLM,TTS;last_error=nonfatal_error:LLM"
    )
    assert lines[2] == "session=test-session turn=2 event=turn_start"


def test_fatal_error_closes_turn_immediately():
    lines = []
    lifecycle = TurnLifecycle("test-session", lines.append)

    lifecycle.start()
    lifecycle.record_error("TTS", fatal=True)

    assert lines[-1] == (
        "session=test-session turn=1 event=turn_incomplete "
        "reason=fatal_pipeline_error;missing_stages=LLM,STT,TTS;last_error=fatal_error:TTS"
    )

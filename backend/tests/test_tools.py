"""Unit tests for Parley insurance mock MCP tools."""

from datetime import date, timedelta
import pytest

from backend.tools.policy_tools import (
    MOCK_POLICIES,
    TOOL_SCHEMAS,
    execute_tool,
    open_claim,
    policy_lookup,
    schedule_callback,
)


# --- Policy Lookup Tests ---

def test_policy_lookup_success():
    res = policy_lookup("POL-4401")
    assert res["success"] is True
    assert res["policy"]["policy_number"] == "POL-4401"
    assert res["policy"]["coverages"]["comprehensive_deductible"] == 500.0
    assert "execution_time_ms" in res
    assert res["execution_time_ms"] < 10.0


def test_policy_lookup_case_insensitive():
    res = policy_lookup("hom-302")
    assert res["success"] is True
    assert res["policy"]["policy_number"] == "HOM-302"
    assert res["policy"]["coverages"]["burst_pipe_freeze_limit"] == 25000.0


def test_policy_lookup_not_found():
    res = policy_lookup("POL-9999")
    assert res["success"] is False
    assert res["error_code"] == "POLICY_NOT_FOUND"
    assert "not found" in res["error"].lower()


def test_policy_lookup_empty_argument():
    res = policy_lookup("")
    assert res["success"] is False
    assert res["error_code"] == "MISSING_ARGUMENT"


# --- Open Claim Tests ---

def test_open_claim_success():
    res = open_claim(
        policy_number="POL-4401",
        incident_type="collision",
        incident_date="2026-02-15",
        description="Rear-ended at a red light on 5th avenue.",
    )
    assert res["success"] is True
    claim = res["claim"]
    assert claim["claim_id"].startswith("CLM-")
    assert claim["policy_number"] == "POL-4401"
    assert claim["status"] == "opened_pending_review"
    assert "Sarah Jenkins" in claim["assigned_adjuster"]
    assert res["execution_time_ms"] < 100.0



def test_open_claim_theft_requires_police_report():
    res = open_claim(
        policy_number="POL-4401",
        incident_type="theft",
        incident_date="2026-03-01",
        description="Catalytic converter stolen overnight while parked.",
    )
    assert res["success"] is True
    reqs = res["claim"]["immediate_requirements"]
    assert any("police report" in r.lower() for r in reqs)


def test_open_claim_unknown_policy():
    res = open_claim(
        policy_number="INVALID-001",
        incident_type="glass_damage",
        incident_date="2026-03-01",
        description="Cracked windshield from rock on highway.",
    )
    assert res["success"] is False
    assert res["error_code"] == "POLICY_NOT_FOUND"


def test_open_claim_invalid_incident_type():
    res = open_claim(
        policy_number="POL-4401",
        incident_type="alien_invasion",
        incident_date="2026-03-01",
        description="UFO landed on vehicle roof.",
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_INCIDENT_TYPE"


def test_open_claim_future_date_rejected():
    tomorrow = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    res = open_claim(
        policy_number="POL-4401",
        incident_type="collision",
        incident_date=tomorrow,
        description="Accident will happen tomorrow.",
    )
    assert res["success"] is False
    assert res["error_code"] == "FUTURE_DATE"


def test_open_claim_invalid_date_format():
    res = open_claim(
        policy_number="POL-4401",
        incident_type="collision",
        incident_date="03/15/2026",
        description="Fender bender in shopping mall lot.",
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_DATE_FORMAT"


def test_open_claim_short_description():
    res = open_claim(
        policy_number="POL-4401",
        incident_type="collision",
        incident_date="2026-02-10",
        description="hit",
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_DESCRIPTION"


# --- Schedule Callback Tests ---

def test_schedule_callback_success():
    res = schedule_callback(
        policy_number="HOM-302",
        preferred_time="Tomorrow 10am-12pm EST",
        phone_number="555-234-5678",
    )
    assert res["success"] is True
    cb = res["callback"]
    assert cb["confirmation_code"].startswith("CB-")
    assert cb["status"] == "confirmed"
    assert "+1 (555) 234-5678" in cb["phone_number"]
    assert res["execution_time_ms"] < 10.0


def test_schedule_callback_invalid_phone():
    res = schedule_callback(
        policy_number="HOM-302",
        preferred_time="Tomorrow 10am",
        phone_number="123",
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_PHONE_NUMBER"


def test_schedule_callback_unknown_policy():
    res = schedule_callback(
        policy_number="BAD-POLICY",
        preferred_time="Morning",
        phone_number="555-345-6789",
    )
    assert res["success"] is False
    assert res["error_code"] == "POLICY_NOT_FOUND"


# --- Tool Dispatcher & Schema Tests ---

def test_execute_tool_dispatcher():
    res = execute_tool("policy_lookup", {"policy_number": "PRP-205"})
    assert res["success"] is True
    assert res["policy"]["policy_number"] == "PRP-205"

    res_unknown = execute_tool("nonexistent_tool", {})
    assert res_unknown["success"] is False
    assert res_unknown["error_code"] == "UNKNOWN_TOOL"


def test_tool_schemas_validity():
    assert len(TOOL_SCHEMAS) == 3
    tool_names = {t["name"] for t in TOOL_SCHEMAS}
    assert tool_names == {"policy_lookup", "open_claim", "schedule_callback"}

    for s in TOOL_SCHEMAS:
        assert "description" in s
        assert "parameters" in s
        assert s["parameters"]["type"] == "object"
        assert len(s["parameters"]["required"]) > 0

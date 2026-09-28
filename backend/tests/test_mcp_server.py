"""Unit tests for Parley MCP stdio server."""

import json
from backend.tools.mcp_server import handle_rpc_request


def test_mcp_initialize():
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    res = handle_rpc_request(req)
    assert res is not None
    assert res["id"] == 1
    assert res["result"]["protocolVersion"] == "2024-11-05"
    assert res["result"]["serverInfo"]["name"] == "parley-policy-tools"
    assert "tools" in res["result"]["capabilities"]


def test_mcp_tools_list():
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    res = handle_rpc_request(req)
    assert res is not None
    tools = res["result"]["tools"]
    assert len(tools) == 3
    tool_names = {t["name"] for t in tools}
    assert tool_names == {"policy_lookup", "open_claim", "schedule_callback"}
    for t in tools:
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_mcp_tools_call_policy_lookup():
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "policy_lookup",
            "arguments": {"policy_number": "POL-4401"},
        },
    }
    res = handle_rpc_request(req)
    assert res is not None
    assert res["id"] == 3
    assert res["result"]["isError"] is False
    content = json.loads(res["result"]["content"][0]["text"])
    assert content["success"] is True
    assert content["policy"]["policy_number"] == "POL-4401"


def test_mcp_tools_call_open_claim():
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "open_claim",
            "arguments": {
                "policy_number": "POL-4401",
                "incident_type": "collision",
                "incident_date": "2026-02-15",
                "description": "Fender bender on 5th Ave.",
            },
        },
    }
    res = handle_rpc_request(req)
    assert res is not None
    assert res["result"]["isError"] is False
    content = json.loads(res["result"]["content"][0]["text"])
    assert content["success"] is True
    assert content["claim"]["claim_id"].startswith("CLM-")


def test_mcp_tools_call_schedule_callback():
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "schedule_callback",
            "arguments": {
                "policy_number": "POL-4401",
                "preferred_time": "Tomorrow morning 10am",
                "phone_number": "555-123-4567",
            },
        },
    }
    res = handle_rpc_request(req)
    assert res is not None
    assert res["result"]["isError"] is False
    content = json.loads(res["result"]["content"][0]["text"])
    assert content["success"] is True
    assert content["callback"]["status"] == "confirmed"


def test_mcp_unknown_method():
    req = {"jsonrpc": "2.0", "id": 6, "method": "unknown_rpc", "params": {}}
    res = handle_rpc_request(req)
    assert res is not None
    assert res["id"] == 6
    assert "error" in res
    assert res["error"]["code"] == -32601


def test_mcp_notification_no_response():
    req = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    res = handle_rpc_request(req)
    assert res is None

"""Model Context Protocol (MCP) server exposing Parley insurance tools over stdio.

Implements JSON-RPC 2.0 protocol for standard MCP clients (Claude Desktop, Cursor, Goose, etc.)
Exposes:
- policy_lookup
- open_claim
- schedule_callback
"""

from __future__ import annotations

import json
import sys
from typing import Any

from backend.tools.policy_tools import TOOL_SCHEMAS, execute_tool

SERVER_NAME = "parley-policy-tools"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


def format_mcp_tools() -> list[dict[str, Any]]:
    """Format TOOL_SCHEMAS into standard MCP tool descriptions."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["parameters"],
        }
        for tool in TOOL_SCHEMAS
    ]


def handle_rpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC 2.0 request or notification."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    # JSON-RPC notifications have no id and require no response
    if req_id is None and method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": format_mcp_tools(),
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        res = execute_tool(tool_name, arguments)
        is_error = not res.get("success", False)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(res, indent=2),
                    }
                ],
                "isError": is_error,
            },
        }

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }


def run_stdio_server():
    """Main stdio loop reading newline-delimited JSON-RPC from sys.stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as err:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {err}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        response = handle_rpc_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()

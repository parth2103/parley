"""Fictional insurance policy & claims mock MCP tools for Parley voice agent.

All data is fictional. No real insurer data or PII.
Compatible with standard function calling / MCP tool schema interfaces.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


# --- Mock In-Memory Database ---

MOCK_POLICIES: dict[str, dict[str, Any]] = {
    "POL-4401": {
        "policy_number": "POL-4401",
        "policy_type": "Auto Comprehensive & Collision",
        "named_insured": "Alex Morgan",
        "status": "active",
        "effective_date": "2026-01-01",
        "expiration_date": "2027-01-01",
        "vehicle": "2024 Subaru Outback (VIN: 4S4BS6CK8R3123456)",
        "coverages": {
            "comprehensive_deductible": 500.0,
            "collision_deductible": 500.0,
            "glass_replacement_deductible": 100.0,
            "glass_chip_repair_deductible": 0.0,
            "towing_limit_per_incident": 150.0,
            "towing_deductible": 50.0,
            "towing_max_miles": 25,
            "rental_reimbursement_daily": 45.0,
            "rental_reimbursement_max_days": 30,
            "rental_reimbursement_max_total": 1350.0,
        },
    },
    "HOM-302": {
        "policy_number": "HOM-302",
        "policy_type": "Homeowners Policy (HO-3)",
        "named_insured": "Alex Morgan",
        "status": "active",
        "effective_date": "2025-06-15",
        "expiration_date": "2026-06-15",
        "property_address": "404 North Elm Street, Springfield, IL 62701",
        "coverages": {
            "dwelling_limit": 350000.0,
            "standard_deductible": 1000.0,
            "burst_pipe_freeze_limit": 25000.0,
            "sewer_backup_limit": 5000.0,
            "sewer_backup_deductible": 1000.0,
            "surface_flood_limit": 0.0,
            "flood_excluded": True,
            "earthquake_excluded": True,
        },
    },
    "PRP-205": {
        "policy_number": "PRP-205",
        "policy_type": "Personal Property & High-Value Rider",
        "named_insured": "Alex Morgan",
        "status": "active",
        "effective_date": "2025-09-01",
        "expiration_date": "2026-09-01",
        "coverages": {
            "off_premises_vehicle_theft_limit": 1500.0,
            "rider_deductible": 250.0,
            "scheduled_electronics_limit": 3000.0,
            "scheduled_electronics_deductible": 100.0,
            "unscheduled_jewelry_limit": 1000.0,
        },
    },
}

VALID_INCIDENT_TYPES = {
    "collision",
    "comprehensive",
    "glass_damage",
    "towing",
    "pipe_freeze",
    "water_backup",
    "theft",
    "vandalism",
    "property_damage",
}

PHONE_REGEX = re.compile(r"^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$")


# --- Tool Implementation Functions ---

def policy_lookup(policy_number: str) -> dict[str, Any]:
    """Look up an existing insurance policy by its identifier.

    Args:
        policy_number: Format like 'POL-4401', 'HOM-302', 'PRP-205'.

    Returns:
        Structured policy details or error payload with execution latency.
    """
    t0 = time.perf_counter()
    p_num = (policy_number or "").strip().upper()

    if not p_num:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "policy_number is required",
            "error_code": "MISSING_ARGUMENT",
            "execution_time_ms": round(elapsed, 3),
        }

    policy = MOCK_POLICIES.get(p_num)
    if not policy:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Policy '{p_num}' not found in active records",
            "error_code": "POLICY_NOT_FOUND",
            "execution_time_ms": round(elapsed, 3),
        }

    # RAG retrieval: fetch relevant policy document clauses from corpus
    from backend.rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retrieved = retriever.retrieve(f"{policy.get('policy_type', '')} {p_num} coverage deductible", top_k=2)
    retrieved_clauses = [f"[{c.chunk_id}] {c.title}: {c.content}" for c, _ in retrieved]

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "success": True,
        "policy": policy,
        "retrieved_clauses": retrieved_clauses,
        "execution_time_ms": round(elapsed, 3),
    }


def open_claim(
    policy_number: str,
    incident_type: str,
    incident_date: str,
    description: str,
) -> dict[str, Any]:
    """Initiate a first notice of loss (FNOL) and open a claim ticket.

    Args:
        policy_number: Active policy identifier.
        incident_type: One of 'collision', 'comprehensive', 'glass_damage',
                       'towing', 'pipe_freeze', 'water_backup', 'theft', 'vandalism'.
        incident_date: Date of incident in 'YYYY-MM-DD' format.
        description: Caller's summary of the damage or event.

    Returns:
        Claim ticket details including claim_id, assigned adjuster, and next steps.
    """
    t0 = time.perf_counter()
    p_num = (policy_number or "").strip().upper()
    i_type = (incident_type or "").strip().lower()
    i_date = (incident_date or "").strip()
    desc = (description or "").strip()

    # 1. Validate policy
    if not p_num:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "policy_number is required",
            "error_code": "MISSING_ARGUMENT",
            "execution_time_ms": round(elapsed, 3),
        }

    policy = MOCK_POLICIES.get(p_num)
    if not policy:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Policy '{p_num}' not found",
            "error_code": "POLICY_NOT_FOUND",
            "execution_time_ms": round(elapsed, 3),
        }

    # 2. Validate incident type
    if not i_type or i_type not in VALID_INCIDENT_TYPES:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Invalid incident_type '{i_type}'. Must be one of: {sorted(list(VALID_INCIDENT_TYPES))}",
            "error_code": "INVALID_INCIDENT_TYPE",
            "execution_time_ms": round(elapsed, 3),
        }

    # 3. Validate incident date
    if not i_date:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "incident_date is required (YYYY-MM-DD)",
            "error_code": "MISSING_ARGUMENT",
            "execution_time_ms": round(elapsed, 3),
        }

    try:
        parsed_date = datetime.strptime(i_date, "%Y-%m-%d").date()
        today = date.today()
        if parsed_date > today:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "success": False,
                "error": f"Incident date '{i_date}' cannot be in the future",
                "error_code": "FUTURE_DATE",
                "execution_time_ms": round(elapsed, 3),
            }
    except ValueError:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Invalid date format '{i_date}'. Required format is YYYY-MM-DD",
            "error_code": "INVALID_DATE_FORMAT",
            "execution_time_ms": round(elapsed, 3),
        }

    # 4. Validate description
    if not desc or len(desc) < 5:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "description must be at least 5 characters long",
            "error_code": "INVALID_DESCRIPTION",
            "execution_time_ms": round(elapsed, 3),
        }

    # Generate deterministic mock claim ID based on policy and date
    claim_hash = abs(hash(f"{p_num}-{i_date}-{i_type}")) % 9000 + 1000
    claim_id = f"CLM-{claim_hash}"

    # Determine next steps based on incident type
    requirements = []
    if i_type in {"theft", "vandalism"}:
        requirements.append("Obtain an official police report within 24 hours and upload the report number.")
    elif i_type == "glass_damage":
        requirements.append("Upload photo of windshield glass damage or schedule certified safelite mobile repair.")
    elif i_type in {"pipe_freeze", "water_backup"}:
        requirements.append("Mitigate further water damage immediately and preserve damaged plumbing parts.")
    else:
        requirements.append("Upload scene photos and repair estimate from an authorized repair shop.")

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "success": True,
        "claim": {
            "claim_id": claim_id,
            "policy_number": p_num,
            "status": "opened_pending_review",
            "incident_type": i_type,
            "incident_date": i_date,
            "assigned_adjuster": "Sarah Jenkins (Senior Claims Representative)",
            "estimated_review_window": "1-2 business days",
            "immediate_requirements": requirements,
        },
        "execution_time_ms": round(elapsed, 3),
    }


def schedule_callback(
    policy_number: str,
    preferred_time: str,
    phone_number: str,
) -> dict[str, Any]:
    """Schedule a dedicated phone callback from a licensed adjuster or agent.

    Args:
        policy_number: Active policy identifier.
        preferred_time: Preferred time window, e.g. 'Morning (9am-12pm)', 'Afternoon (1pm-5pm)'.
        phone_number: 10-digit telephone contact number.

    Returns:
        Callback confirmation details including confirmation_code and scheduled window.
    """
    t0 = time.perf_counter()
    p_num = (policy_number or "").strip().upper()
    p_time = (preferred_time or "").strip()
    raw_phone = (phone_number or "").strip()

    # 1. Validate policy number
    if not p_num:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "policy_number is required",
            "error_code": "MISSING_ARGUMENT",
            "execution_time_ms": round(elapsed, 3),
        }

    policy = MOCK_POLICIES.get(p_num)
    if not policy:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Policy '{p_num}' not found",
            "error_code": "POLICY_NOT_FOUND",
            "execution_time_ms": round(elapsed, 3),
        }

    # 2. Validate phone number
    match = PHONE_REGEX.match(raw_phone)
    if not match:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"Invalid telephone number '{raw_phone}'. Must be a valid 10-digit US phone number",
            "error_code": "INVALID_PHONE_NUMBER",
            "execution_time_ms": round(elapsed, 3),
        }

    formatted_phone = f"+1 ({match.group(1)}) {match.group(2)}-{match.group(3)}"

    # 3. Validate preferred time
    if not p_time:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": "preferred_time is required",
            "error_code": "MISSING_ARGUMENT",
            "execution_time_ms": round(elapsed, 3),
        }

    cb_hash = abs(hash(f"{p_num}-{raw_phone}-{p_time}")) % 9000 + 1000
    confirmation_code = f"CB-{cb_hash}"

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "success": True,
        "callback": {
            "confirmation_code": confirmation_code,
            "policy_number": p_num,
            "phone_number": formatted_phone,
            "scheduled_window": p_time,
            "status": "confirmed",
            "notice": "An adjuster will call this number during your scheduled window.",
        },
        "execution_time_ms": round(elapsed, 3),
    }


# --- Tool Schemas for Function Calling & MCP ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "policy_lookup",
        "description": "Look up active insurance policy details, deductible amounts, coverage limits, and policy clauses by policy number.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_number": {
                    "type": "string",
                    "description": "The insurance policy number (e.g. 'POL-4401', 'HOM-302', 'PRP-205').",
                },
            },
            "required": ["policy_number"],
        },
    },
    {
        "name": "open_claim",
        "description": "File a first notice of loss (FNOL) to open a formal insurance claim ticket for accident, damage, water, or theft.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_number": {
                    "type": "string",
                    "description": "The active policy number under which the claim is being filed.",
                },
                "incident_type": {
                    "type": "string",
                    "enum": list(sorted(VALID_INCIDENT_TYPES)),
                    "description": "The category of incident (e.g. 'collision', 'comprehensive', 'glass_damage', 'theft', 'pipe_freeze').",
                },
                "incident_date": {
                    "type": "string",
                    "description": "The date the incident occurred in ISO YYYY-MM-DD format.",
                },
                "description": {
                    "type": "string",
                    "description": "A concise summary of how the damage occurred and the items affected.",
                },
            },
            "required": ["policy_number", "incident_type", "incident_date", "description"],
        },
    },
    {
        "name": "schedule_callback",
        "description": "Schedule a telephone callback appointment with an insurance adjuster or claims specialist.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_number": {
                    "type": "string",
                    "description": "The caller's policy identifier.",
                },
                "preferred_time": {
                    "type": "string",
                    "description": "The caller's preferred callback time window (e.g. 'Tomorrow Morning 9am-11am', 'Friday Afternoon 2pm-4pm').",
                },
                "phone_number": {
                    "type": "string",
                    "description": "The 10-digit telephone number to receive the callback.",
                },
            },
            "required": ["policy_number", "preferred_time", "phone_number"],
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatcher for executing tools dynamically by name with schema validation."""
    if name == "policy_lookup":
        return policy_lookup(policy_number=arguments.get("policy_number", ""))
    elif name == "open_claim":
        return open_claim(
            policy_number=arguments.get("policy_number", ""),
            incident_type=arguments.get("incident_type", ""),
            incident_date=arguments.get("incident_date", ""),
            description=arguments.get("description", ""),
        )
    elif name == "schedule_callback":
        return schedule_callback(
            policy_number=arguments.get("policy_number", ""),
            preferred_time=arguments.get("preferred_time", ""),
            phone_number=arguments.get("phone_number", ""),
        )
    else:
        return {
            "success": False,
            "error": f"Unknown tool: '{name}'",
            "error_code": "UNKNOWN_TOOL",
            "execution_time_ms": 0.0,
        }


def get_policy_function_schemas():
    """Return FunctionSchema list with async handlers attached for Pipecat LLMContext."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    async def _async_tool_handler(params):
        fn_name = params.function_name
        args = dict(params.arguments)
        res = execute_tool(fn_name, args)
        await params.result_callback(res)

    return [
        FunctionSchema(
            name=tool["name"],
            description=tool["description"],
            properties=tool["parameters"]["properties"],
            required=tool["parameters"]["required"],
            handler=_async_tool_handler,
        )
        for tool in TOOL_SCHEMAS
    ]


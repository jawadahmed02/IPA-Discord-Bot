import json
import os
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


DEFAULT_PAAS_MCP_URL = "https://solver.planning.domains/mcp"


# Retrieves the PaaS MCP URL from environment variables, with a default fallback
def paas_mcp_url() -> str:
    return os.getenv("PAAS_MCP_URL", DEFAULT_PAAS_MCP_URL).strip() or DEFAULT_PAAS_MCP_URL

# Normalizes MCP tool results into plain text that Discord can display
def _tool_text(result: Any) -> str:
    if getattr(result, "isError", False):
        if getattr(result, "content", None):
            return "\n".join(
                item.text for item in result.content if getattr(item, "text", None)
            ).strip() or "Tool returned an error."
        return "Tool returned an error."

    if not getattr(result, "content", None):
        return ""

    texts = [item.text for item in result.content if getattr(item, "text", None)]
    return "\n".join(texts).strip()


# Extracts the planner's actual plan text from the MCP JSON payload when present
def _extract_plan_text(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    if not isinstance(payload, dict):
        return text.strip()

    output = payload.get("output")
    if isinstance(output, dict):
        sas_plan = output.get("sas_plan")
        if isinstance(sas_plan, str) and sas_plan.strip():
            return sas_plan.strip()

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    return text.strip()


# Opens an MCP session, calls one named PaaS tool, and returns its text output
async def _call_paas_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    async with streamable_http_client(paas_mcp_url()) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _tool_text(result)


# Minimal bot-facing entrypoint for solving one domain/problem pair through PaaS
async def solve_pddl(domain: str, problem: str, timeout_s: int = 30) -> str:
    result = await _call_paas_tool(
        "paas_lama_first_solve",
        {"domain": domain, "problem": problem, "timeout_s": timeout_s},
    )
    return _extract_plan_text(result)

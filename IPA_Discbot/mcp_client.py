import json
import os
import shlex
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


# Allows the bot to pass optional environment overrides into the spawned MCP server
def _env_json(name: str) -> dict[str, str] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} must decode to an object")

    return {str(k): str(v) for k, v in parsed.items()}


# Centralizes how we launch the local PaaS MCP wrapper over stdio
def paas_stdio_server() -> StdioServerParameters:
    command = os.getenv("PAAS_MCP_COMMAND", "python3").strip() or "python3"
    args_raw = os.getenv("PAAS_MCP_ARGS", "").strip()
    args = shlex.split(args_raw, posix=False) if args_raw else []

    if not args:
        raise RuntimeError(
            "Missing PAAS_MCP_ARGS. Configure the planning-as-a-service MCP wrapper in your .env, "
            "for example on Windows:\n"
            'PAAS_MCP_COMMAND=py\n'
            'PAAS_MCP_ARGS="-3 C:/path/to/planning-as-a-service/server/mcp/mcp_wrap.py"\n'
            'PAAS_MCP_ENV_JSON={"PAAS_HOST":"localhost","PAAS_PORT":"5001"}'
        )

    env = _env_json("PAAS_MCP_ENV_JSON")
    return StdioServerParameters(command=command, args=args, env=env)


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
    server = paas_stdio_server()

    async with stdio_client(server) as (read, write):
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

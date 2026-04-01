from typing import Any

from .config import (
    L2P_DOMAIN_TOOL,
    L2P_TASK_TOOL,
    MCPServerName,
    PAAS_SOLVE_TOOL,
    PAAS_VAL_TOOL,
    PAAS_VALIDATE_DOMAIN_TOOL,
    PAAS_VALIDATE_PLAN_TOOL,
    PAAS_VALIDATE_TASK_TOOL,
    mcp_server_url,
)
from .manager import call_mcp_tool, list_mcp_tools
from .parsing import (
    compact_tool_arguments,
    extract_val_text,
    default_expected_tools,
    extract_plan_text,
    format_tool_list,
    require_dict_payload,
)

_MCP_TOOL_CATALOG: dict[MCPServerName, list[dict[str, Any]]] = {
    "paas": [],
    "l2p": [],
}


async def solve_pddl(domain: str, problem: str, timeout_s: int = 30) -> str:
    result = await call_mcp_tool(
        "paas",
        PAAS_SOLVE_TOOL,
        {"domain": domain, "problem": problem, "timeout_s": timeout_s},
    )
    return extract_plan_text(result)


async def validate_domain(
    domain: str,
    timeout_s: int = 30,
) -> Any:
    result = await call_mcp_tool(
        "paas",
        PAAS_VALIDATE_DOMAIN_TOOL,
        {
            "domain": domain,
            "timeout_s": timeout_s,
        },
    )
    return result


async def validate_plan(
    domain: str,
    problem: str,
    plan: str,
    timeout_s: int = 30,
) -> Any:
    result = await call_mcp_tool(
        "paas",
        PAAS_VALIDATE_PLAN_TOOL,
        {
            "domain": domain,
            "problem": problem,
            "plan": plan,
            "timeout_s": timeout_s,
        },
    )
    return result


async def validate_task(
    domain: str,
    problem: str,
    timeout_s: int = 30,
) -> Any:
    result = await call_mcp_tool(
        "paas",
        PAAS_VALIDATE_TASK_TOOL,
        {
            "domain": domain,
            "problem": problem,
            "timeout_s": timeout_s,
        },
    )
    return result


async def validate_plan_with_val(
    domain: str,
    problem: str,
    plan: str,
    timeout_s: int = 30,
) -> str:
    result = await call_mcp_tool(
        "paas",
        PAAS_VAL_TOOL,
        {
            "domain": domain,
            "problem": problem,
            "plan": plan,
            "timeout_s": timeout_s,
        },
    )
    return extract_val_text(result)


async def update_domain_via_l2p(
    *,
    domain_update: str | None = None,
    domain_update_path: str | None = None,
    domain: dict[str, Any] | None = None,
    domain_name: str | None = None,
    action_name: str | list[str] | None = None,
    replace_fields: list[str] | None = None,
    use_type_hierarchy: bool = False,
    infer_requirements: bool = True,
) -> dict[str, Any]:
    result = await call_mcp_tool(
        "l2p",
        L2P_DOMAIN_TOOL,
        compact_tool_arguments(
            {
                "domain_update": domain_update,
                "domain_update_path": domain_update_path,
                "domain": domain,
                "domain_name": domain_name,
                "action_name": action_name,
                "replace_fields": replace_fields,
                "use_type_hierarchy": use_type_hierarchy,
                "infer_requirements": infer_requirements,
            }
        ),
    )
    return require_dict_payload(L2P_DOMAIN_TOOL, result)


async def update_task_via_l2p(
    *,
    task_update: str | None = None,
    task_update_path: str | None = None,
    task: dict[str, Any] | None = None,
    domain_name: str | None = None,
    problem_name: str | None = None,
    replace_fields: list[str] | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    result = await call_mcp_tool(
        "l2p",
        L2P_TASK_TOOL,
        compact_tool_arguments(
            {
                "task_update": task_update,
                "task_update_path": task_update_path,
                "task": task,
                "domain_name": domain_name,
                "problem_name": problem_name,
                "replace_fields": replace_fields,
                "metric": metric,
            }
        ),
    )
    return require_dict_payload(L2P_TASK_TOOL, result)


async def verify_mcp_server(
    server: MCPServerName,
    expected_tools: tuple[str, ...] | None = None,
) -> str:
    required_tools = expected_tools or default_expected_tools(server)
    tool_names = [tool["name"] for tool in await list_mcp_tools(server)]
    missing_tools = [tool for tool in required_tools if tool not in tool_names]
    if missing_tools:
        available = ", ".join(tool_names) or "none"
        raise RuntimeError(
            f"Connected to MCP server at {mcp_server_url(server)}, but it did not advertise "
            f"{format_tool_list(tuple(missing_tools))}. Available tools: {available}"
        )
    return (
        f"Connected to {mcp_server_url(server)} and found "
        f"{format_tool_list(required_tools)}."
    )


async def verify_remote_mcp_server(expected_tool: str = PAAS_SOLVE_TOOL) -> str:
    return await verify_mcp_server("paas", (expected_tool,))


async def verify_l2p_mcp_server() -> str:
    return await verify_mcp_server("l2p", (L2P_DOMAIN_TOOL, L2P_TASK_TOOL))


async def list_all_mcp_tools() -> dict[MCPServerName, list[dict[str, str]]]:
    return {
        "paas": await list_mcp_tools("paas"),
        "l2p": await list_mcp_tools("l2p"),
    }


async def refresh_mcp_tool_catalog() -> dict[MCPServerName, list[dict[str, Any]]]:
    global _MCP_TOOL_CATALOG
    _MCP_TOOL_CATALOG = {
        "paas": [dict(tool) for tool in await list_mcp_tools("paas")],
        "l2p": [dict(tool) for tool in await list_mcp_tools("l2p")],
    }
    return {
        server: [dict(tool) for tool in tools]
        for server, tools in _MCP_TOOL_CATALOG.items()
    }


async def get_mcp_tool_catalog() -> dict[MCPServerName, list[dict[str, Any]]]:
    if any(_MCP_TOOL_CATALOG[server] for server in _MCP_TOOL_CATALOG):
        return {
            server: [dict(tool) for tool in tools]
            for server, tools in _MCP_TOOL_CATALOG.items()
        }
    return await refresh_mcp_tool_catalog()

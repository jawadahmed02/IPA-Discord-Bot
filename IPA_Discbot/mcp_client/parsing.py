import json
from typing import Any

from .config import L2P_DOMAIN_TOOL, L2P_TASK_TOOL, MCPServerName, PAAS_SOLVE_TOOL


def tool_text(result: Any) -> str:
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


def tool_payload(result: Any) -> Any:
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is not None:
        return structured_content
    return tool_text(result)


def compact_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if value is not None}


def _extract_plan_from_mapping(mapping: dict[str, Any]) -> str | None:
    output = mapping.get("output")
    if isinstance(output, dict):
        sas_plan = output.get("sas_plan")
        if isinstance(sas_plan, str) and sas_plan.strip():
            return sas_plan.strip()

    nested_result = mapping.get("result")
    if isinstance(nested_result, dict):
        nested_text = _extract_plan_from_mapping(nested_result)
        if nested_text:
            return nested_text

    error = mapping.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    return None


def extract_plan_text(payload: Any) -> str:
    if isinstance(payload, dict):
        extracted = _extract_plan_from_mapping(payload)
        return extracted or json.dumps(payload).strip()

    text = str(payload)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    if not isinstance(decoded, dict):
        return text.strip()

    extracted = _extract_plan_from_mapping(decoded)
    return extracted or text.strip()


def require_dict_payload(tool_name: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"`{tool_name}` returned a non-JSON response: {payload.strip() or '<empty>'}"
            ) from exc
        if isinstance(decoded, dict):
            return decoded
    raise RuntimeError(
        f"`{tool_name}` returned a non-JSON response: {str(payload).strip() or '<empty>'}"
    )


def format_tool_list(tool_names: tuple[str, ...]) -> str:
    names = [f"`{tool_name}`" for tool_name in tool_names]
    if not names:
        return "no tools"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def default_expected_tools(server: MCPServerName) -> tuple[str, ...]:
    if server == "l2p":
        return (L2P_DOMAIN_TOOL, L2P_TASK_TOOL)
    return (PAAS_SOLVE_TOOL,)

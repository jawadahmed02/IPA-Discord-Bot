import asyncio
import json
import re
import threading

import discord
import llm

from IPA_Discbot.mcp_client import (
    L2P_DOMAIN_TOOL,
    L2P_TASK_TOOL,
    get_mcp_tool_catalog,
    parse_solve_response_text as _parse_solve_response_text,
)
from .config import MODEL
from .storage import get_effective_provider_key, get_user_model


def _build_transcript(context_messages: list[dict]) -> str:
    lines: list[str] = []
    for m in context_messages:
        role = m.get("role", "").strip().lower()
        content = (m.get("content") or "").strip()

        if role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = "User"

        lines.append(f"{prefix}: {content}")

    lines.append("Assistant:")

    return "\n".join(lines)


def _conversation_system_prompt() -> str:
    return (
        "You are a helpful planning assistant inside Discord. "
        "Keep answers concise. "
        "You are given persisted conversation history for this user, and that history can include previous bot sessions. "
        "Use facts, variables, preferences, and prior commitments from the provided history when answering. "
        "If the needed information appears in the provided history, do not say you cannot access previous sessions or that memory was lost after a restart. "
        "Only say information is unavailable when it truly does not appear in the supplied conversation history."
    )


def _parse_llm_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start : end + 1])


def _json_repair_system_prompt() -> str:
    return (
        "You repair malformed JSON. "
        "Return exactly one valid JSON object and nothing else. "
        "Preserve the original intended fields and values when possible."
    )


def _join_natural(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _describe_plan_action(action_name: str, args: list[str]) -> str:
    tokens = [token for token in action_name.split("_") if token]
    if not tokens:
        return "Perform an action."

    prepositions = {"in", "into", "on", "onto", "from", "to", "at", "with", "using", "via"}
    split_index = next((index for index, token in enumerate(tokens) if token in prepositions), None)

    if split_index is None:
        verb_phrase = " ".join(tokens).capitalize()
        if not args:
            return verb_phrase + "."
        return f"{verb_phrase}: {_join_natural(args)}."

    before = tokens[:split_index]
    prep = tokens[split_index]
    after = tokens[split_index + 1 :]

    subject_text = " ".join(before).capitalize() if before else "Do"
    object_hint = " ".join(after)

    if len(args) == 0:
        if object_hint:
            return f"{subject_text} {prep} {object_hint}."
        return f"{subject_text}."

    if len(args) == 1:
        if object_hint:
            return f"{subject_text} {args[0]} {prep} {object_hint}."
        return f"{subject_text} {prep} {args[0]}."

    main_args = args[:-1]
    target_arg = args[-1]
    if object_hint:
        return f"{subject_text} {_join_natural(main_args)} {prep} {target_arg} ({object_hint})."
    return f"{subject_text} {_join_natural(main_args)} {prep} {target_arg}."


def _plan_to_natural_language(plan_text: str) -> str:
    steps: list[str] = []
    step_number = 1

    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if not (line.startswith("(") and line.endswith(")")):
            continue

        parts = line[1:-1].split()
        if not parts:
            continue

        action_name = parts[0]
        args = parts[1:]
        steps.append(f"{step_number}. {_describe_plan_action(action_name, args)}")
        step_number += 1

    if not steps:
        return "Natural-language steps unavailable."

    return "Natural-language steps:\n" + "\n".join(steps)


def _to_pddl_identifier(value: str, default: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned or default


def _fallback_natural_language_solve_system_prompt() -> str:
    return (
        "Return only valid JSON with keys "
        "`domain_name`, `problem_name`, `action_name`, `domain_update`, and `task_update`. "
        "Use `domain_update` and `task_update` in the exact heading/block format expected by the planning server."
    )


async def _natural_language_solve_system_prompt() -> str:
    try:
        tool_catalog = await get_mcp_tool_catalog()
    except Exception:
        return _fallback_natural_language_solve_system_prompt()

    l2p_tools = {
        str(tool.get("name", "")).strip(): tool for tool in tool_catalog.get("l2p", [])
    }
    domain_description = str(
        l2p_tools.get(L2P_DOMAIN_TOOL, {}).get("description", "")
    ).strip()
    task_description = str(
        l2p_tools.get(L2P_TASK_TOOL, {}).get("description", "")
    ).strip()

    if not domain_description or not task_description:
        return _fallback_natural_language_solve_system_prompt()

    return (
        "You convert a user's natural-language planning request into two planning-server updates. "
        "Return only valid JSON with this schema: "
        "{\"domain_name\":\"snake_case_name\",\"problem_name\":\"snake_case_name\","
        "\"action_name\":[\"snake_case_action\"],"
        "\"domain_update\":\"text formatted for update_domain\","
        "\"task_update\":\"text formatted for update_task\"}. "
        "Split the user's request into domain rules/operators and task-specific objects, initial state, and goal. "
        "Choose concise snake_case names. "
        "For scheduling or time-allocation problems, preserve time windows and AM/PM distinctions explicitly in durations, free/busy slots, and object names. "
        "Do not model one-slot tasks with unused extra slot parameters in the plan representation. "
        "The `domain_update` field must follow the `update_domain` tool description exactly:\n"
        f"{domain_description}\n\n"
        "The `task_update` field must follow the `update_task` tool description exactly:\n"
        f"{task_description}\n\n"
        "Do not include markdown outside the JSON object."
    )


def _llm_reply_sync(model_id: str, context_messages: list[dict]) -> str:
    transcript = _build_transcript(context_messages)
    model = llm.get_model(model_id)
    response = model.prompt(transcript, system=_conversation_system_prompt())
    return response.text().strip()


# One lock per model ID keeps concurrent requests from different users from racing
# on the same model instance's .key attribute.
_MODEL_LOCKS: dict[str, threading.Lock] = {}
_MODEL_LOCKS_LOCK = threading.Lock()


def _get_model_lock(model_id: str) -> threading.Lock:
    with _MODEL_LOCKS_LOCK:
        if model_id not in _MODEL_LOCKS:
            _MODEL_LOCKS[model_id] = threading.Lock()
        return _MODEL_LOCKS[model_id]


def _provider_from_model_id(model_id: str) -> str | None:
    mid = model_id.lower()
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        return "ollama"
    if "claude" in mid or mid.startswith("anthropic"):
        return "anthropic"
    if "gemini" in mid:
        return "gemini"
    return "openai"


def _resolve_user_key(user_id: str, model_id: str) -> str | None:
    provider = _provider_from_model_id(model_id)
    if provider == "ollama":
        return None
    user_key, _ = get_effective_provider_key(user_id, provider)
    if not user_key:
        raise RuntimeError(
            f"No API key available for {provider}. Use /setkey {provider} <key>, "
            "or have someone enable !share for that provider."
        )
    return user_key


def _call_model_sync(model_id: str, user_key: str | None, prompt: str, system_prompt: str) -> str:
    lock = _get_model_lock(model_id)
    with lock:
        model = llm.get_model(model_id)
        original_key = getattr(model, "key", None)
        try:
            if user_key is not None:
                model.key = user_key
            response = model.prompt(prompt, system=system_prompt)
            return response.text().strip()
        finally:
            model.key = original_key


def _run_llm_prompt_for_user_sync(
    user_id: str, model_id: str, prompt: str, system_prompt: str
) -> str:
    user_key = _resolve_user_key(user_id, model_id)
    return _call_model_sync(model_id, user_key, prompt, system_prompt)


def _run_llm_for_user_sync(
    user_id: str, model_id: str, context_messages: list[dict]
) -> str:
    user_key = _resolve_user_key(user_id, model_id)
    transcript = _build_transcript(context_messages)
    return _call_model_sync(model_id, user_key, transcript, _conversation_system_prompt())


async def llm_reply(
    model_id: str, context_messages: list[dict], user_id: str | None = None
) -> str:
    if user_id is None:
        return await asyncio.to_thread(_llm_reply_sync, model_id, context_messages)
    return await asyncio.to_thread(
        _run_llm_for_user_sync, user_id, model_id, context_messages
    )


async def _request_llm_json(message: discord.Message, prompt: str, system_prompt: str) -> dict:
    selected_model = get_user_model(str(message.author.id)) or MODEL
    raw = await asyncio.to_thread(
        _run_llm_prompt_for_user_sync,
        str(message.author.id),
        selected_model,
        prompt,
        system_prompt,
    )
    try:
        return _parse_llm_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        repair_prompt = (
            "Repair this malformed JSON output so it becomes one valid JSON object.\n"
            f"Malformed output: {raw}"
        )
        repaired_raw = await asyncio.to_thread(
            _run_llm_prompt_for_user_sync,
            str(message.author.id),
            selected_model,
            repair_prompt,
            _json_repair_system_prompt(),
        )
        return _parse_llm_json_object(repaired_raw)


def _normalize_server_update_payload(data: dict) -> dict:
    action_name = data.get("action_name")
    if isinstance(action_name, str):
        normalized_action_name: str | list[str] | None = _to_pddl_identifier(action_name, "action")
    elif isinstance(action_name, list):
        normalized_action_name = [
            _to_pddl_identifier(str(item), f"action_{index}")
            for index, item in enumerate(action_name, start=1)
            if str(item).strip()
        ]
    else:
        normalized_action_name = None

    return {
        "domain_name": _to_pddl_identifier(str(data.get("domain_name", "")).strip(), "generated_domain"),
        "problem_name": _to_pddl_identifier(str(data.get("problem_name", "")).strip(), "generated_problem"),
        "action_name": normalized_action_name,
        "domain_update": str(data.get("domain_update", "")).strip(),
        "task_update": str(data.get("task_update", "")).strip(),
    }


async def _llm_plan_from_natural_language(
    message: discord.Message, request_text: str, feedback: str | None = None
) -> dict:
    prompt = (
        "Convert this planning request into one domain update and one problem update for the local MCP planning server.\n"
        "Return only JSON with keys domain_name, problem_name, action_name, domain_update, and task_update.\n"
        f"Original request: {json.dumps(request_text)}"
    )
    if feedback:
        prompt += (
            "\nThe previous attempt failed after being sent to the planning tools. "
            "Rewrite the full JSON so the server can parse it and the generated PDDL is valid.\n"
            f"Planner feedback: {json.dumps(feedback)}"
        )
    data = await _request_llm_json(
        message, prompt, await _natural_language_solve_system_prompt()
    )
    normalized = _normalize_server_update_payload(data)

    if not normalized["domain_update"] or not normalized["task_update"]:
        raise RuntimeError("The model did not return both `domain_update` and `task_update`.")

    return normalized


async def _llm_classify_confirmation_reply(message: discord.Message) -> str:
    prompt = (
        "Classify the user's reply to a confirmation question.\n"
        "Return only JSON with this schema:\n"
        '{"reply_type":"confirm|reject|cancel|other"}\n'
        f"User reply: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify short confirmation replies for a Discord bot. "
        "Infer intent from meaning, not fixed keywords. "
        "Return only valid JSON."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    reply_type = str(data.get("reply_type", "other")).strip().lower()
    if reply_type not in {"confirm", "reject", "cancel", "other"}:
        return "other"
    return reply_type


async def _llm_classify_member_request(message: discord.Message) -> dict:
    mentioned_users = [
        {
            "id": user.id,
            "name": getattr(user, "display_name", None) or user.name,
        }
        for user in message.mentions
    ]
    prompt = (
        "Decide whether this Discord message is asking the bot to mention a server member or add mentioned users to a thread.\n"
        "Return only JSON with this schema:\n"
        '{"intent":"member_lookup|thread_add_mentions|none","requested_name":"string","should_add_to_thread":true}\n'
        f"Is thread: {json.dumps(isinstance(message.channel, discord.Thread))}\n"
        f"Mentioned users: {json.dumps(mentioned_users)}\n"
        f"Message: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify Discord bot control requests. "
        "Infer intent from the user's wording, even when synonyms or unusual phrasing are used. "
        "Use thread_add_mentions only when the message is asking to add already-mentioned users to the current thread. "
        "Use member_lookup when the user refers to someone by name and the bot should resolve and mention them. "
        "Set should_add_to_thread true when the user wants the matched person brought into the current thread. "
        "Return only valid JSON."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    intent = str(data.get("intent", "none")).strip().lower()
    if intent not in {"member_lookup", "thread_add_mentions", "none"}:
        intent = "none"

    requested_name = str(data.get("requested_name", "") or "").strip()
    should_add_to_thread = bool(data.get("should_add_to_thread", False))
    return {
        "intent": intent,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }


async def _llm_classify_workflow_request(message: discord.Message) -> str:
    prompt = (
        "Classify this Discord message for a planning bot workflow.\n"
        "Return only JSON with this schema:\n"
        '{"intent":"plan|domain|problem|validate_plan|validate_domain|validate_task|tools|help|chat"}\n'
        f"Message: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify whether a user is asking a planning bot to perform a tool-backed workflow or just chat. "
        "Use `plan` when the user wants a plan or asks the bot to solve a planning task. "
        "Use `domain` when the user wants a domain generated. "
        "Use `problem` when the user wants a problem/task generated. "
        "Use `validate_plan` when the user wants a plan checked or validated. "
        "Use `validate_domain` when the user wants only a domain checked. "
        "Use `validate_task` when the user wants a domain/problem pair checked. "
        "Use `tools` when the user asks what tools are available. "
        "Use `help` when the user asks what commands or capabilities exist. "
        "Use `chat` for normal conversation, explanation, or anything that should not invoke a workflow. "
        "Return only valid JSON."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    intent = str(data.get("intent", "chat")).strip().lower()
    if intent not in {
        "plan",
        "domain",
        "problem",
        "validate_plan",
        "validate_domain",
        "validate_task",
        "tools",
        "help",
        "chat",
    }:
        return "chat"
    return intent


async def _edit_domain_system_prompt() -> str:
    try:
        tool_catalog = await get_mcp_tool_catalog()
    except Exception:
        return (
            "Return only valid JSON with keys `domain_name`, `action_name`, and `domain_update`. "
            "Use `domain_update` in the exact heading/block format expected by update_domain."
        )

    l2p_tools = {
        str(tool.get("name", "")).strip(): tool for tool in tool_catalog.get("l2p", [])
    }
    domain_description = str(
        l2p_tools.get(L2P_DOMAIN_TOOL, {}).get("description", "")
    ).strip()
    if not domain_description:
        return (
            "Return only valid JSON with keys `domain_name`, `action_name`, and `domain_update`. "
            "Use `domain_update` in the exact heading/block format expected by update_domain."
        )

    return (
        "You revise an existing planning domain according to a user's edit request. "
        "Preserve the user's original intent and keep changes minimal. "
        "Do not rewrite unrelated parts of the domain. "
        "Return only valid JSON with this schema: "
        "{\"domain_name\":\"snake_case_name\",\"action_name\":[\"snake_case_action\"],"
        "\"domain_update\":\"text formatted for update_domain\"}. "
        "The `domain_update` field must follow the `update_domain` tool description exactly:\n"
        f"{domain_description}\n\n"
        "Do not include markdown outside the JSON object."
    )


async def _edit_problem_system_prompt() -> str:
    try:
        tool_catalog = await get_mcp_tool_catalog()
    except Exception:
        return (
            "Return only valid JSON with keys `domain_name`, `problem_name`, and `task_update`. "
            "Use `task_update` in the exact heading/block format expected by update_task."
        )

    l2p_tools = {
        str(tool.get("name", "")).strip(): tool for tool in tool_catalog.get("l2p", [])
    }
    task_description = str(
        l2p_tools.get(L2P_TASK_TOOL, {}).get("description", "")
    ).strip()
    if not task_description:
        return (
            "Return only valid JSON with keys `domain_name`, `problem_name`, and `task_update`. "
            "Use `task_update` in the exact heading/block format expected by update_task."
        )

    return (
        "You revise an existing planning problem according to a user's edit request. "
        "Preserve the user's original intent and keep changes minimal. "
        "Do not rewrite unrelated parts of the problem. "
        "Always include a complete, valid problem update with objects, initial state, and goal state information. "
        "Never omit the goal section, even if it stays unchanged. "
        "Return only valid JSON with this schema: "
        "{\"domain_name\":\"snake_case_name\",\"problem_name\":\"snake_case_name\","
        "\"task_update\":\"text formatted for update_task\"}. "
        "The `task_update` field must follow the `update_task` tool description exactly:\n"
        f"{task_description}\n\n"
        "Do not include markdown outside the JSON object."
    )


async def _llm_domain_edit_from_instruction(
    message: discord.Message,
    instruction: str,
    current_domain: str,
    domain_name: str,
    feedback: str | None = None,
) -> dict:
    prompt = (
        "Revise this planning domain according to the user's instruction.\n"
        "Return only JSON with keys domain_name, action_name, and domain_update.\n"
        f"Current domain name: {json.dumps(domain_name)}\n"
        f"Edit instruction: {json.dumps(instruction)}\n"
        f"Current domain PDDL:\n{current_domain}"
    )
    if feedback:
        prompt += (
            "\nThe previous attempt failed after being sent to the planning tool. "
            "Revise only what is necessary and keep the rest intact.\n"
            f"Tool feedback: {json.dumps(feedback)}"
        )
    data = await _request_llm_json(message, prompt, await _edit_domain_system_prompt())
    normalized = _normalize_server_update_payload(
        {
            "domain_name": data.get("domain_name") or domain_name,
            "problem_name": "unused_problem",
            "action_name": data.get("action_name"),
            "domain_update": data.get("domain_update"),
            "task_update": "unused",
        }
    )
    if not normalized["domain_update"]:
        raise RuntimeError("The model did not return `domain_update`.")
    return normalized


async def _llm_problem_edit_from_instruction(
    message: discord.Message,
    instruction: str,
    current_problem: str,
    domain_name: str,
    problem_name: str,
    feedback: str | None = None,
) -> dict:
    prompt = (
        "Revise this planning problem according to the user's instruction.\n"
        "Return only JSON with keys domain_name, problem_name, and task_update.\n"
        "Keep the plan fixed and preserve unchanged objects, initial state facts, and goal facts unless the instruction or feedback requires a minimal change.\n"
        "The revised task_update must remain complete and explicitly include the goal information.\n"
        f"Current domain name: {json.dumps(domain_name)}\n"
        f"Current problem name: {json.dumps(problem_name)}\n"
        f"Edit instruction: {json.dumps(instruction)}\n"
        f"Current problem PDDL:\n{current_problem}"
    )
    if feedback:
        prompt += (
            "\nThe previous attempt failed after being sent to the planning tool. "
            "Revise only what is necessary and keep the rest intact.\n"
            f"Tool feedback: {json.dumps(feedback)}"
        )
    data = await _request_llm_json(message, prompt, await _edit_problem_system_prompt())
    normalized = _normalize_server_update_payload(
        {
            "domain_name": data.get("domain_name") or domain_name,
            "problem_name": data.get("problem_name") or problem_name,
            "action_name": None,
            "domain_update": "unused",
            "task_update": data.get("task_update"),
        }
    )
    if not normalized["task_update"]:
        raise RuntimeError("The model did not return `task_update`.")
    return normalized


def _clean_pddl_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


async def _llm_domain_pddl_edit_from_instruction(
    message: discord.Message,
    instruction: str,
    current_domain: str,
    domain_name: str,
    feedback: str | None = None,
) -> dict:
    prompt = (
        "Revise this planning domain PDDL according to the user's instruction.\n"
        "Return only JSON with this schema:\n"
        '{"domain_name":"snake_case_name","domain_pddl":"complete revised domain PDDL"}\n'
        "Preserve the user's original intent and keep changes minimal.\n"
        "Return the full revised domain PDDL, not a diff and not update instructions.\n"
        f"Current domain name: {json.dumps(domain_name)}\n"
        f"Edit instruction: {json.dumps(instruction)}\n"
        f"Current domain PDDL:\n{current_domain}"
    )
    if feedback:
        prompt += (
            "\nThe previous revised domain was invalid. "
            "Fix only what is necessary and return one complete valid domain PDDL document.\n"
            f"Validation feedback: {json.dumps(feedback)}"
        )
    system_prompt = (
        "You edit planning domain PDDL directly. "
        "Return only valid JSON. "
        "The `domain_pddl` value must be the complete revised domain file with no markdown fences and no commentary."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    revised_name = _to_pddl_identifier(str(data.get("domain_name") or domain_name).strip(), domain_name or "domain")
    domain_pddl = _clean_pddl_text(str(data.get("domain_pddl", "")))
    if not domain_pddl:
        raise RuntimeError("The model did not return revised domain PDDL.")
    return {"domain_name": revised_name, "domain_pddl": domain_pddl}


async def _llm_problem_pddl_edit_from_instruction(
    message: discord.Message,
    instruction: str,
    current_domain: str,
    current_problem: str,
    domain_name: str,
    problem_name: str,
    feedback: str | None = None,
) -> dict:
    prompt = (
        "Revise this planning problem PDDL according to the user's instruction.\n"
        "Return only JSON with this schema:\n"
        '{"domain_name":"snake_case_name","problem_name":"snake_case_name","problem_pddl":"complete revised problem PDDL"}\n'
        "Preserve the user's original intent and keep changes minimal.\n"
        "Return the full revised problem PDDL, not a diff and not update instructions.\n"
        "Keep object names, initial facts, and goal facts unchanged unless the instruction requires a targeted edit.\n"
        f"Current domain name: {json.dumps(domain_name)}\n"
        f"Current problem name: {json.dumps(problem_name)}\n"
        f"Edit instruction: {json.dumps(instruction)}\n"
        f"Current domain PDDL:\n{current_domain}\n\n"
        f"Current problem PDDL:\n{current_problem}"
    )
    if feedback:
        prompt += (
            "\nThe previous revised problem was invalid. "
            "Fix only what is necessary and return one complete valid problem PDDL document.\n"
            f"Validation feedback: {json.dumps(feedback)}"
        )
    system_prompt = (
        "You edit planning problem PDDL directly. "
        "Return only valid JSON. "
        "The `problem_pddl` value must be the complete revised problem file with no markdown fences and no commentary."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    revised_domain_name = _to_pddl_identifier(
        str(data.get("domain_name") or domain_name).strip(), domain_name or "domain"
    )
    revised_problem_name = _to_pddl_identifier(
        str(data.get("problem_name") or problem_name).strip(), problem_name or "problem"
    )
    problem_pddl = _clean_pddl_text(str(data.get("problem_pddl", "")))
    if not problem_pddl:
        raise RuntimeError("The model did not return revised problem PDDL.")
    return {
        "domain_name": revised_domain_name,
        "problem_name": revised_problem_name,
        "problem_pddl": problem_pddl,
    }


async def _llm_plan_edit_from_instruction(
    message: discord.Message,
    instruction: str,
    current_plan: str,
) -> str:
    prompt = (
        "Revise this plan according to the user's instruction.\n"
        "Return only JSON with this schema:\n"
        '{"plan":"one action per line"}\n'
        "Preserve unaffected steps and keep changes minimal.\n"
        f"Edit instruction: {json.dumps(instruction)}\n"
        f"Current plan:\n{current_plan}"
    )
    system_prompt = (
        "You edit planning plans. Return only valid JSON. "
        "The `plan` value must contain only the revised plan text, typically one action per line, with no markdown fences."
    )
    data = await _request_llm_json(message, prompt, system_prompt)
    plan_text = str(data.get("plan", "")).strip()
    if not plan_text:
        raise RuntimeError("The model did not return a revised plan.")
    return plan_text


async def _llm_explain_artifact(
    message: discord.Message,
    artifact_type: str,
    artifact_text: str,
) -> str:
    prompt = (
        f"Give a short, general explanation of what this planning {artifact_type} does in normal natural language.\n"
        "Keep it high level and easy to follow.\n"
        "Do not go into too much detail unless it is necessary to understand the main idea.\n"
        "Do not restate the full PDDL verbatim.\n"
        "Prefer a brief paragraph or a few short bullets.\n"
        f"{artifact_type.capitalize()} content:\n{artifact_text}"
    )
    system_prompt = (
        "You explain planning artifacts for users who may not know PDDL. "
        "Be accurate, concise, and high level. "
        "Focus on the main purpose and behavior, not line-by-line details."
    )
    selected_model = get_user_model(str(message.author.id)) or MODEL
    return (
        await asyncio.to_thread(
            _run_llm_prompt_for_user_sync,
            str(message.author.id),
            selected_model,
            prompt,
            system_prompt,
        )
    ).strip()


def _all_llm_model_ids() -> list[str]:
    return sorted([m.model_id for m in llm.get_models()])

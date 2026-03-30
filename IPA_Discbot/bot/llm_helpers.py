import asyncio
import json
import os
import re
import threading

import discord
import llm

from ..mcp_client import L2P_DOMAIN_TOOL, L2P_TASK_TOOL, get_mcp_tool_catalog
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


def _parse_solve_response_text(text: str) -> str:
    try:
        payload = _parse_llm_json_object(text)
    except ValueError:
        return text.strip()
    except json.JSONDecodeError:
        return text.strip()

    output = payload.get("output")
    if isinstance(output, dict):
        sas_plan = output.get("sas_plan")
        if isinstance(sas_plan, str) and sas_plan.strip():
            return sas_plan.strip()

    result = payload.get("result")
    if isinstance(result, dict):
        result_output = result.get("output")
        if isinstance(result_output, dict):
            sas_plan = result_output.get("sas_plan")
            if isinstance(sas_plan, str) and sas_plan.strip():
                return sas_plan.strip()

        result_error = result.get("error")
        if isinstance(result_error, str) and result_error.strip():
            return result_error.strip()

        result_stdout = result.get("stdout")
        if isinstance(result_stdout, str) and result_stdout.strip():
            return result_stdout.strip()

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    stdout = payload.get("stdout")
    if isinstance(stdout, str) and stdout.strip():
        return stdout.strip()

    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw_result = raw.get("result")
        if isinstance(raw_result, dict):
            raw_stdout = raw_result.get("stdout")
            if isinstance(raw_stdout, str) and raw_stdout.strip():
                return raw_stdout.strip()

    return text.strip()


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


def _format_solve_reply(plan_text: str) -> str:
    readable = _plan_to_natural_language(plan_text)
    return f"{readable}\n\nRaw plan:\n```text\n{plan_text.strip()}\n```"


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


LLM_ENV_LOCK = threading.Lock()

PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "LLM_GEMINI_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def _provider_from_model_id(model_id: str) -> str | None:
    mid = model_id.lower()
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        return "ollama"
    if "claude" in mid or mid.startswith("anthropic"):
        return "anthropic"
    if "gemini" in mid:
        return "gemini"
    return "openai"


def _run_llm_prompt_for_user_sync(
    user_id: str, model_id: str, prompt: str, system_prompt: str
) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    user_key = None
    if env_key:
        user_key, _ = get_effective_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(
                f"No API key available for {provider}. Use /setkey {provider} <key>, "
                "or have someone enable !share for that provider."
            )

    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)
            response = model.prompt(prompt, system=system_prompt)
            return response.text().strip()
        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


def _run_llm_for_user_sync(
    user_id: str, model_id: str, context_messages: list[dict]
) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    transcript = _build_transcript(context_messages)
    system_prompt = _conversation_system_prompt()

    user_key = None
    if env_key:
        user_key, _ = get_effective_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(
                f"No API key available for {provider}. Use /setkey {provider} <key>, "
                "or have someone enable !share for that provider."
            )

    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)
            resp = model.prompt(transcript, system=system_prompt)
            return resp.text().strip()
        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


async def llm_reply(
    model_id: str, context_messages: list[dict], user_id: str | None = None
) -> str:
    print("========== MODEL DEBUG ==========")
    print("User ID:", user_id)
    print("DB model:", get_user_model(user_id) if user_id else None)
    print("Passed model_id:", model_id)
    print("Fallback MODEL constant:", MODEL)
    print("=================================")

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


def _all_llm_model_ids() -> list[str]:
    return sorted([m.model_id for m in llm.get_models()])

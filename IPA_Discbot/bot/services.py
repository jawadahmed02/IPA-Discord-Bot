import asyncio
import ast
import difflib
import io
import json
import re
import traceback

import discord
from discord import app_commands
from discord.ext import commands

from IPA_Discbot.mcp_client import (
    close_mcp_servers,
    connect_mcp_servers,
    list_all_mcp_tools,
    get_mcp_tool_catalog,
    solve_pddl,
    update_domain_via_l2p,
    update_task_via_l2p,
    validate_plan_with_val,
    validate_domain,
    validate_plan,
    validate_task,
)
from .llm_helpers import (
    _all_llm_model_ids,
    _llm_classify_workflow_request,
    _llm_domain_pddl_edit_from_instruction,
    _llm_domain_edit_from_instruction,
    _llm_explain_artifact,
    _format_solve_reply,
    _llm_classify_confirmation_reply,
    _llm_classify_member_request,
    _llm_plan_edit_from_instruction,
    _llm_problem_pddl_edit_from_instruction,
    _llm_plan_from_natural_language,
    _llm_problem_edit_from_instruction,
    _parse_solve_response_text,
    llm_reply,
)
from .config import (
    ARTIFACT_HISTORY,
    BOT_SESSION_ID,
    GUILD_ID,
    LAST_SOLVE_ARTIFACTS,
    MODEL,
    PENDING_MEMBER_CONFIRMATIONS,
    bot,
)
from .storage import (
    get_share_mode,
    get_recent_context,
    get_user_model,
    is_chat_enabled,
    load_saved_working_artifacts,
    log_message,
    save_current_session,
    save_working_artifacts_snapshot,
    save_provider_key,
    set_chat_enabled,
    set_share_mode,
    set_user_model,
    user_has_any_provider_key,
)


def _truncate_discord_message(text: str, limit: int = 1900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _split_discord_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]

    fenced_match = re.match(r"^([\s\S]*?)```([a-zA-Z0-9_-]*)\n([\s\S]*?)\n```$", text)
    if fenced_match:
        prefix = fenced_match.group(1).rstrip()
        info = fenced_match.group(2)
        body = fenced_match.group(3)
        fence_open = f"```{info}\n"
        fence_close = "\n```"

        chunks: list[str] = []
        body_lines = body.splitlines()
        current_body: list[str] = []
        current_prefix = prefix

        for line in body_lines:
            candidate_body = "\n".join(current_body + [line])
            candidate = (
                (current_prefix + "\n" if current_prefix else "")
                + fence_open
                + candidate_body
                + fence_close
            )
            if current_body and len(candidate) > limit:
                completed = (
                    (current_prefix + "\n" if current_prefix else "")
                    + fence_open
                    + "\n".join(current_body)
                    + fence_close
                )
                chunks.append(completed)
                current_body = [line]
                current_prefix = ""
            else:
                current_body.append(line)

        if current_body:
            completed = (
                (current_prefix + "\n" if current_prefix else "")
                + fence_open
                + "\n".join(current_body)
                + fence_close
            )
            chunks.append(completed)

        if chunks:
            return chunks

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in text.splitlines():
        addition = len(line) + (1 if current else 0)
        if current and current_length + addition > limit:
            chunks.append("\n".join(current))
            current = [line]
            current_length = len(line)
        else:
            current.append(line)
            current_length += addition

    if current:
        chunks.append("\n".join(current))

    return chunks


def _summarize_tool_description(description: str, limit: int = 80) -> str:
    summary = " ".join((description or "").split())
    if not summary:
        return "No description provided."
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3].rstrip() + "..."


def _format_mcp_tools_message(tool_map: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []

    for server_name in ("paas", "l2p"):
        lines.append(f"{server_name.upper()} tools:")
        tools = tool_map.get(server_name, [])
        if not tools:
            lines.append("- none")
        for tool in tools:
            name = tool.get("name", "unknown-tool")
            description = _summarize_tool_description(tool.get("description", ""))
            lines.append(f"- `{name}`: {description}")
        lines.append("")

    return "\n".join(lines).strip()


def _format_single_server_tools_message(
    server_name: str, tools: list[dict[str, str]]
) -> str:
    lines = [f"{server_name.upper()} tools:"]
    if not tools:
        lines.append("- none")
        return "\n".join(lines)

    for tool in tools:
        name = tool.get("name", "unknown-tool")
        description = _summarize_tool_description(tool.get("description", ""))
        lines.append(f"- `{name}`: {description}")

    return "\n".join(lines)


def _format_help_message() -> str:
    lines = [
        "Available bot commands:",
        "",
        "Planning:",
        "`!plan <request>` Solve a planning request from natural language, or attach domain/problem PDDL files with `!plan`.",
        "`!plan` Re-solve using the current stored domain and problem in this channel for you.",
        "`!help` Show this help message with a short description of each command.",
        "`!domain <request>` Generate a planning domain from a natural-language request and return the domain PDDL.",
        "`!problem <request>` Generate a planning problem from a natural-language request and return the problem PDDL.",
        "",
        "HITL Editing:",
        "`!show <domain|problem|plan>` Show the current working artifact and return it as a file.",
        "`!files` Send the current stored domain, problem, and plan as files.",
        "`!explain <domain|problem|plan>` Explain the current working artifact in normal language.",
        "`!edit <domain|problem|plan> <instruction>` Revise one current artifact while preserving the rest of the workflow state.",
        "`!undo <domain|problem|plan>` Restore the previous version of one artifact.",
        "",
        "Validation:",
        "`!validate` Validate a plan against a domain and problem, using attachments or the last successful `!plan` output.",
        "`!validate_domain` Validate a domain PDDL file with the PaaS domain validation tool.",
        "`!validate_plan` Validate a domain/problem/plan triple with the PaaS plan validation tool.",
        "`!validate_task` Validate a domain/problem pair with the PaaS task validation tool.",
        "",
        "Progress / Beta:",
        "`!autovalidate` Beta: loop domain/problem repairs against VAL until the current plan passes or the retry limit is reached.",
        "",
        "MCP Tools:",
        "`!paastools` List only the tools currently exposed by the connected PaaS MCP server.",
        "`!tools` List the MCP tools currently exposed by the connected planning servers.",
        "",
        "Chat and Threads:",
        "`!thread [topic]` Create a private thread in the current server channel for chatting with the bot.",
        "`!chat` Toggle normal-message bot chat on or off for you in the current channel.",
        "",
        "Bot Settings:",
        "`!share` Toggle whether other users may use your saved provider key when they do not have one.",
        "`!save` Save the current bot session so its conversation log is kept after restart.",
        "`/models` Show the available LLM model IDs you can choose from.",
        "`/use <model_id>` Set which model the bot should use for your requests.",
        "`/setkey <provider> <api_key>` Save your provider API key for `openai`, `gemini`, or `anthropic`.",
    ]
    return "\n".join(lines)


def _pddl_from_l2p_payload(payload: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_pddl_reply(kind: str, name: str, text: str) -> str:
    title = f"Generated {kind} `{name}`:" if name else f"Generated {kind}:"
    stripped = text.strip()
    fence = "lisp" if stripped.startswith("(define") else "text"
    return f"{title}\n```{fence}\n{stripped}\n```"


def _format_updated_pddl_reply(kind: str, name: str, text: str) -> str:
    title = f"Updated {kind} `{name}`:" if name else f"Updated {kind}:"
    stripped = text.strip()
    fence = "lisp" if stripped.startswith("(define") else "text"
    return f"{title}\n```{fence}\n{stripped}\n```"


def _format_domain_reply(domain_name: str, domain_text: str) -> str:
    return _format_pddl_reply("domain", domain_name, domain_text)


def _format_problem_reply(problem_name: str, problem_text: str) -> str:
    return _format_pddl_reply("problem", problem_name, problem_text)


def _safe_pddl_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_")
    return cleaned or fallback


def _text_file(text: str, filename: str) -> discord.File:
    return discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)


def _solve_output_has_action_steps(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("(") and line.endswith(")"):
            return True
    return False


def _planner_output_indicates_failure(text: str) -> bool:
    lowered = text.lower()
    failure_markers = (
        "driver aborting",
        "translate exit code",
        "search exit code",
        "planner failed",
        "unsolvable",
        "error:",
    )
    return any(marker in lowered for marker in failure_markers)


def _parse_requested_grid_shape(request_text: str) -> tuple[int, int] | None:
    text = (request_text or "").lower()
    match = re.search(r"\b(\d+)\s*x\s*(\d+)\s+grid\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"\b(\d+)\s+by\s+(\d+)\s+grid\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def _request_mentions_top_left_start(request_text: str) -> bool:
    text = (request_text or "").lower()
    return "top left" in text and ("start" in text or "starting" in text or "begin" in text)


def _request_mentions_bottom_right_goal(request_text: str) -> bool:
    text = (request_text or "").lower()
    return "bottom right" in text


def _problem_contains_cell(problem_text: str, row: int, col: int) -> bool:
    return re.search(rf"\bcell_{row}_{col}\b", problem_text) is not None


def _problem_sets_start_cell(problem_text: str, row: int, col: int) -> bool:
    return re.search(rf"\(at\s+cell_{row}_{col}\)", problem_text) is not None


def _problem_sets_goal_cell(problem_text: str, row: int, col: int) -> bool:
    goal_match = re.search(r"\(:goal\b(.*?)\)\s*\)\s*$", problem_text, re.IGNORECASE | re.DOTALL)
    if goal_match is None:
        goal_match = re.search(r"\(:goal\b(.*)", problem_text, re.IGNORECASE | re.DOTALL)
    goal_block = goal_match.group(1) if goal_match else problem_text
    return re.search(rf"\(at\s+cell_{row}_{col}\)", goal_block) is not None


def _problem_uses_out_of_bounds_cell(problem_text: str, rows: int, cols: int) -> bool:
    for row_text, col_text in re.findall(r"\bcell_(\d+)_(\d+)\b", problem_text):
        row = int(row_text)
        col = int(col_text)
        if row >= rows or col >= cols:
            return True
    return False


def _check_request_matches_generated_problem(request_text: str, problem_text: str) -> str | None:
    grid_shape = _parse_requested_grid_shape(request_text)
    if grid_shape is not None:
        rows, cols = grid_shape
        max_row = rows - 1
        max_col = cols - 1
        if not _problem_contains_cell(problem_text, max_row, max_col):
            return (
                f"The request asked for a {rows}x{cols} grid, so the generated problem must include "
                f"`cell_{max_row}_{max_col}`. It does not."
            )
        if _problem_uses_out_of_bounds_cell(problem_text, rows, cols):
            return "The generated problem uses cell indices outside the requested grid size."
        if _request_mentions_top_left_start(request_text) and not _problem_sets_start_cell(problem_text, 0, 0):
            return "The request starts at the top-left corner, so the initial state must place the agent at `cell_0_0`."
        if _request_mentions_bottom_right_goal(request_text) and not _problem_sets_goal_cell(problem_text, max_row, max_col):
            return (
                f"The request targets the bottom-right corner of a {rows}x{cols} grid, "
                f"so the goal must be `cell_{max_row}_{max_col}`."
            )
    return None


def _extract_plan_steps(plan_text: str) -> list[list[str]]:
    steps: list[list[str]] = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if not (line.startswith("(") and line.endswith(")")):
            continue
        tokens = line[1:-1].split()
        if tokens:
            steps.append(tokens)
    return steps


def _check_request_matches_generated_plan(request_text: str, plan_text: str) -> str | None:
    grid_shape = _parse_requested_grid_shape(request_text)
    if grid_shape is None:
        return None

    rows, cols = grid_shape
    max_row = rows - 1
    max_col = cols - 1
    steps = _extract_plan_steps(plan_text)
    if not steps:
        return "The generated plan did not contain any recognizable action steps."

    if _request_mentions_top_left_start(request_text):
        first_args = steps[0][1:]
        if not first_args or first_args[0] != "cell_0_0":
            return "The request starts at the top-left corner, so the plan must start from `cell_0_0`."

    if _request_mentions_bottom_right_goal(request_text):
        last_args = steps[-1][1:]
        if len(last_args) < 2 or last_args[1] != f"cell_{max_row}_{max_col}":
            return (
                f"The request targets the bottom-right corner of a {rows}x{cols} grid, "
                f"so the final move must end at `cell_{max_row}_{max_col}`."
            )

    minimum_steps = (rows - 1) + (cols - 1)
    if len(steps) < minimum_steps:
        return (
            f"A top-left to bottom-right path on a {rows}x{cols} grid cannot be shorter than "
            f"{minimum_steps} moves, but the generated plan has only {len(steps)}."
        )

    return None


async def _read_text_attachment(attachment: discord.Attachment) -> str:
    data = await attachment.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"{attachment.filename} is not valid UTF-8 text.") from e


async def _extract_pddl_attachments(message: discord.Message) -> tuple[str, str]:
    attachments = list(message.attachments)
    if len(attachments) < 2:
        raise RuntimeError("Attach both a domain PDDL file and a problem PDDL file.")

    domain_text: str | None = None
    problem_text: str | None = None

    for attachment in attachments:
        filename = attachment.filename.lower()
        content = await _read_text_attachment(attachment)

        if domain_text is None and "domain" in filename:
            domain_text = content
            continue

        if problem_text is None and "problem" in filename:
            problem_text = content

    if domain_text is None or problem_text is None:
        pddl_files = [
            attachment
            for attachment in attachments
            if attachment.filename.lower().endswith((".pddl", ".txt"))
        ]
        if len(pddl_files) >= 2:
            if domain_text is None:
                domain_text = await _read_text_attachment(pddl_files[0])
            if problem_text is None:
                problem_text = await _read_text_attachment(pddl_files[1])

    if domain_text is None or problem_text is None:
        raise RuntimeError(
            "Could not identify both files. Name them with `domain` and `problem`, "
            "or attach exactly two PDDL text files."
        )

    return domain_text, problem_text


async def _extract_val_attachments(message: discord.Message) -> tuple[str, str, str]:
    attachments = list(message.attachments)
    if len(attachments) < 3:
        raise RuntimeError("Attach a domain file, a problem file, and a plan file.")

    domain_text: str | None = None
    problem_text: str | None = None
    plan_text: str | None = None

    for attachment in attachments:
        filename = attachment.filename.lower()
        content = await _read_text_attachment(attachment)

        if domain_text is None and "domain" in filename:
            domain_text = content
            continue

        if problem_text is None and "problem" in filename:
            problem_text = content
            continue

        if plan_text is None and "plan" in filename:
            plan_text = content

    if domain_text is None or problem_text is None or plan_text is None:
        text_files = [
            attachment
            for attachment in attachments
            if attachment.filename.lower().endswith((".pddl", ".txt", ".plan", ".sol"))
        ]
        if len(text_files) >= 3:
            contents: dict[str, str] = {}
            for attachment in text_files[:3]:
                contents[attachment.filename] = await _read_text_attachment(attachment)

            ordered_contents = list(contents.values())
            if domain_text is None:
                domain_text = ordered_contents[0]
            if problem_text is None:
                problem_text = ordered_contents[1]
            if plan_text is None:
                plan_text = ordered_contents[2]

    if domain_text is None or problem_text is None or plan_text is None:
        raise RuntimeError(
            "Could not identify all three files. Name them with `domain`, `problem`, and `plan`, "
            "or attach exactly three text files in that order."
        )

    return domain_text, problem_text, plan_text


async def _extract_domain_attachment(message: discord.Message) -> str:
    attachments = list(message.attachments)
    if not attachments:
        raise RuntimeError("Attach a domain PDDL file.")

    for attachment in attachments:
        filename = attachment.filename.lower()
        if "domain" in filename:
            return await _read_text_attachment(attachment)

    text_files = [
        attachment
        for attachment in attachments
        if attachment.filename.lower().endswith((".pddl", ".txt"))
    ]
    if text_files:
        return await _read_text_attachment(text_files[0])

    raise RuntimeError("Could not identify a domain file.")


def _solve_artifacts_key(message: discord.Message) -> tuple[int, int]:
    return (message.channel.id, message.author.id)


def _latest_artifact_key_for_user(user_id: int) -> tuple[int, int] | None:
    for key in reversed(list(LAST_SOLVE_ARTIFACTS.keys())):
        if key[1] != user_id:
            continue
        artifacts = LAST_SOLVE_ARTIFACTS.get(key, {})
        if any(str(artifacts.get(name, "")).strip() for name in ("domain", "problem", "plan")):
            return key
    return None


def _working_artifacts(message: discord.Message) -> dict[str, str]:
    key = _solve_artifacts_key(message)
    current = LAST_SOLVE_ARTIFACTS.get(key)
    if current is not None:
        return current

    fallback_key = _latest_artifact_key_for_user(message.author.id)
    if fallback_key is not None and fallback_key != key:
        LAST_SOLVE_ARTIFACTS[key] = {
            k: str(v) for k, v in LAST_SOLVE_ARTIFACTS.get(fallback_key, {}).items()
        }
        fallback_history = ARTIFACT_HISTORY.get(fallback_key, [])
        if fallback_history:
            ARTIFACT_HISTORY[key] = [
                {k: str(v) for k, v in snapshot.items()} for snapshot in fallback_history
            ]
        return LAST_SOLVE_ARTIFACTS[key]

    return LAST_SOLVE_ARTIFACTS.setdefault(key, {})


def _persist_artifacts_if_session_saved() -> None:
    save_working_artifacts_snapshot(
        BOT_SESSION_ID,
        LAST_SOLVE_ARTIFACTS,
        ARTIFACT_HISTORY,
    )


def _push_artifact_history(message: discord.Message) -> None:
    key = _solve_artifacts_key(message)
    current = LAST_SOLVE_ARTIFACTS.get(key)
    if not current:
        return
    snapshot = {k: str(v) for k, v in current.items()}
    history = ARTIFACT_HISTORY.setdefault(key, [])
    history.append(snapshot)
    if len(history) > 10:
        del history[:-10]


def _update_working_artifacts(message: discord.Message, **updates: str) -> dict[str, str]:
    current = _working_artifacts(message)
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        value_str = str(value)
        if current.get(key) != value_str:
            changed = True
    if changed and current:
        _push_artifact_history(message)
    current.update({k: str(v) for k, v in updates.items() if v is not None})
    if changed:
        _persist_artifacts_if_session_saved()
    return current


def _artifact_text(current: dict[str, str], artifact_type: str) -> str:
    return str(current.get(artifact_type, "")).strip()


def _artifact_filename(current: dict[str, str], artifact_type: str) -> str:
    if artifact_type == "domain":
        return f"{_safe_pddl_name(current.get('domain_name', ''), 'domain')}.pddl"
    if artifact_type == "problem":
        return f"{_safe_pddl_name(current.get('problem_name', ''), 'problem')}.pddl"
    return "plan.txt"


def _artifact_reply_text(current: dict[str, str], artifact_type: str) -> str:
    text = _artifact_text(current, artifact_type)
    if not text:
        return ""
    if artifact_type == "domain":
        return _format_domain_reply(str(current.get("domain_name", "")).strip(), text)
    if artifact_type == "problem":
        return _format_problem_reply(str(current.get("problem_name", "")).strip(), text)
    return f"Current plan:\n```text\n{text}\n```"


def _val_output_indicates_valid(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    positive_markers = (
        "plan valid",
        "plan executed successfully",
        "successful plans:",
        "plan verification result: valid",
        "plan verification result: success",
        "plan successfully validated",
    )
    negative_markers = (
        "error:",
        "unknown type",
        "type-checking",
        "goal not satisfied",
        "unsatisfied",
        "invalid",
        "failed",
        "problem in domain definition",
        "problem in problem definition",
        "problem in plan definition",
        "precondition",
        "cannot be applied",
        "violat",
    )
    if any(marker in lowered for marker in negative_markers):
        return False
    return any(marker in lowered for marker in positive_markers)


def _parse_loose_structured_text(text: str) -> dict | None:
    stripped = (text or "").strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            decoded = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return None
    return decoded if isinstance(decoded, dict) else None


def _extract_validation_payload(raw: object) -> dict:
    if isinstance(raw, dict):
        payload = raw
    else:
        payload = _parse_loose_structured_text(str(raw)) or {}

    while isinstance(payload.get("result"), dict):
        payload = payload["result"]
    return payload


def _collect_validation_text(payload: dict) -> str:
    texts: list[str] = []

    def _add(value: object) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped not in texts:
                texts.append(stripped)

    output = payload.get("output")
    if isinstance(output, dict):
        for key in ("val.log", "pddl_domain.log", "pddl_problem.log", "pddl_plan.log", "stdout", "stderr"):
            _add(output.get(key))
        for key, value in output.items():
            if key not in {"val.log", "pddl_domain.log", "pddl_problem.log", "pddl_plan.log", "stdout", "stderr"}:
                _add(value)
    else:
        _add(output)

    for key in ("stdout", "stderr", "error"):
        _add(payload.get(key))

    raw = payload.get("raw")
    if isinstance(raw, dict):
        nested = raw.get("result")
        if isinstance(nested, dict):
            for key in ("output", "stdout", "stderr", "error"):
                value = nested.get(key)
                if isinstance(value, dict):
                    for subkey in ("val.log", "pddl_domain.log", "pddl_problem.log", "pddl_plan.log", "stdout", "stderr"):
                        _add(value.get(subkey))
                    for _, subvalue in value.items():
                        _add(subvalue)
                else:
                    _add(value)

    return "\n\n".join(texts).strip()


def _summarize_validation_failure(details: str) -> str:
    text = (details or "").strip()
    if not text:
        return ""

    priority_patterns = (
        r"pddl\.[A-Za-z0-9_.]*Error:\s*[^\n]+",
        r"[A-Za-z0-9_.]*Error:\s*[^\n]+",
        r"problem in (?:domain|problem|plan) definition[^\n]*",
        r"unknown type[^\n]*",
        r"type-?checking[^\n]*",
        r"goal not satisfied[^\n]*",
        r"cannot be applied[^\n]*",
        r"precondition[^\n]*",
    )
    for pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("("):
            continue
        if stripped.startswith("File "):
            continue
        if stripped.lower().startswith("traceback"):
            continue
        if stripped.startswith("```"):
            continue
        return stripped

    return ""


def _validation_kind_labels(kind: str) -> tuple[str, str]:
    if kind == "domain":
        return ("Domain", "domain")
    if kind == "task":
        return ("Task", "task")
    return ("Plan", "plan")


def _validation_indicates_valid(kind: str, raw: object) -> bool:
    payload = _extract_validation_payload(raw)
    text = _collect_validation_text(payload) or str(raw or "")
    lowered = text.lower()

    common_negative_markers = (
        "error:",
        "unknown type",
        "type-checking",
        "invalid",
        "failed",
        "problem in domain definition",
        "problem in problem definition",
        "problem in plan definition",
        "violat",
        "timed out",
        "timeout",
    )
    plan_negative_markers = (
        "goal not satisfied",
        "unsatisfied",
        "precondition",
        "cannot be applied",
    )

    negative_markers = common_negative_markers + (plan_negative_markers if kind == "plan" else ())
    if any(marker in lowered for marker in negative_markers):
        return False

    status = str(payload.get("status", "")).strip().lower()
    if status == "ok":
        return True

    if kind == "plan":
        return _val_output_indicates_valid(text)
    return False


def _format_validation_result(kind: str, raw: object) -> str:
    title, noun = _validation_kind_labels(kind)
    payload = _extract_validation_payload(raw)
    details = _collect_validation_text(payload)
    check_url = str(payload.get("check_url", "")).strip()
    is_valid = _validation_indicates_valid(kind, raw)
    failure_summary = _summarize_validation_failure(details) if not is_valid else ""

    if is_valid:
        message = f"{title} validation passed."
    else:
        message = f"{title} validation failed."
        if failure_summary:
            message += f"\n\nReason: `{failure_summary}`"

    if details:
        message += f"\n\nDetails:\n```text\n{details}\n```"
    elif not is_valid:
        fallback = str(raw).strip()
        if fallback:
            message += f"\n\nDetails:\n```text\n{fallback}\n```"

    if check_url:
        message += f"\n\nCheck URL: {check_url}"

    return _truncate_discord_message(message)


def _store_last_solve_artifacts(
    message: discord.Message,
    domain_text: str,
    problem_text: str,
    plan_text: str,
    domain_name: str,
    problem_name: str,
    **extra_artifacts: str,
) -> None:
    artifacts = {
        "domain": domain_text,
        "problem": problem_text,
        "plan": plan_text,
        "domain_name": domain_name,
        "problem_name": problem_name,
    }
    artifacts.update({k: str(v) for k, v in extra_artifacts.items() if v is not None})
    LAST_SOLVE_ARTIFACTS[_solve_artifacts_key(message)] = artifacts
    _persist_artifacts_if_session_saved()


async def _update_domain_with_fallback(
    *,
    domain_update: str,
    domain_name: str,
    action_name: str | list[str] | None,
) -> dict[str, object]:
    try:
        return await update_domain_via_l2p(
            domain_update=domain_update,
            domain_name=domain_name,
            action_name=action_name,
        )
    except RuntimeError as e:
        message = str(e)
        if "More action_name values were provided than action updates" not in message:
            raise
        return await update_domain_via_l2p(
            domain_update=domain_update,
            domain_name=domain_name,
        )


async def _run_plan_request(
    message: discord.Message,
    request_text: str | None = None,
) -> tuple[str, list[discord.File]]:
    domain_name = "domain"
    problem_name = "problem"

    if message.attachments:
        domain_text, problem_text = await _extract_pddl_attachments(message)
        result = await solve_pddl(domain_text, problem_text)
        result = _parse_solve_response_text(result)
    else:
        request_text = (request_text or "").strip()
        if not request_text:
            current = _working_artifacts(message)
            domain_text = _artifact_text(current, "domain")
            problem_text = _artifact_text(current, "problem")
            domain_name = str(current.get("domain_name", domain_name)).strip() or domain_name
            problem_name = str(current.get("problem_name", problem_name)).strip() or problem_name

            if not domain_text and not problem_text:
                raise RuntimeError(
                    "No current domain or problem to solve. Run `!domain`, `!problem`, or `!plan` first, or attach domain and problem PDDL files."
                )
            if not domain_text:
                raise RuntimeError(
                    "No current domain to solve with. Run `!domain` or `!plan` first, or attach a domain PDDL file."
                )
            if not problem_text:
                raise RuntimeError(
                    "No current problem to solve with. Run `!problem` or `!plan` first, or attach a problem PDDL file."
                )

            raw_result = await solve_pddl(domain_text, problem_text)
            result = _parse_solve_response_text(raw_result)
            if _planner_output_indicates_failure(result) and not _solve_output_has_action_steps(
                result
            ):
                raise RuntimeError(result or "Failed to produce a valid plan from the current domain and problem.")
        else:
            result = ""
            retry_feedback: str | None = None

            for _ in range(2):
                llm_plan = await _llm_plan_from_natural_language(
                    message, request_text, retry_feedback
                )
                domain_name = str(llm_plan.get("domain_name", "")).strip()
                problem_name = str(llm_plan.get("problem_name", "")).strip()
                domain_update = str(llm_plan.get("domain_update", "")).strip()
                task_update = str(llm_plan.get("task_update", "")).strip()
                action_name = llm_plan.get("action_name")

                if not domain_name or not problem_name or not domain_update or not task_update:
                    raise RuntimeError("The model did not return a complete domain/task update payload.")

                try:
                    domain_payload = await _update_domain_with_fallback(
                        domain_update=domain_update,
                        domain_name=domain_name,
                        action_name=action_name,
                    )
                    task_payload = await update_task_via_l2p(
                        task_update=task_update,
                        domain_name=domain_name,
                        problem_name=problem_name,
                    )
                except RuntimeError as e:
                    retry_feedback = str(e)
                    continue

                domain_text = _pddl_from_l2p_payload(
                    domain_payload,
                    "domain_pddl",
                    "domain",
                    "pddl",
                )
                problem_text = _pddl_from_l2p_payload(
                    task_payload,
                    "task_pddl",
                    "problem_pddl",
                    "problem",
                    "task",
                    "pddl",
                )

                if not domain_text or not problem_text:
                    raise RuntimeError("Failed to generate PDDL from the natural-language request.")

                request_mismatch = _check_request_matches_generated_problem(
                    request_text,
                    problem_text,
                )
                if request_mismatch:
                    retry_feedback = (
                        "The generated problem did not faithfully match the user's request. "
                        f"{request_mismatch}"
                    )
                    continue

                raw_result = await solve_pddl(domain_text, problem_text)
                result = _parse_solve_response_text(raw_result)
                if _planner_output_indicates_failure(result) and not _solve_output_has_action_steps(
                    result
                ):
                    retry_feedback = result
                    continue
                plan_mismatch = _check_request_matches_generated_plan(
                    request_text,
                    result,
                )
                if plan_mismatch:
                    retry_feedback = (
                        "The generated plan did not faithfully match the user's request. "
                        f"{plan_mismatch}"
                    )
                    continue
                break
            else:
                raise RuntimeError(retry_feedback or "Failed to produce a valid plan.")

    _store_last_solve_artifacts(
        message, domain_text, problem_text, result, domain_name, problem_name
    )
    return f"Generated planning artifacts.\n\nCurrent plan:\n```text\n{result.strip()}\n```", None


async def _run_domain_request(message: discord.Message, request_text: str) -> tuple[str, str]:
    retry_feedback: str | None = None
    domain_name = ""
    domain_text = ""

    for _ in range(2):
        llm_plan = await _llm_plan_from_natural_language(
            message, request_text, retry_feedback
        )
        domain_name = str(llm_plan.get("domain_name", "")).strip()
        domain_update = str(llm_plan.get("domain_update", "")).strip()
        action_name = llm_plan.get("action_name")

        if not domain_name or not domain_update:
            raise RuntimeError("The model did not return a complete domain update payload.")

        try:
            domain_payload = await _update_domain_with_fallback(
                domain_update=domain_update,
                domain_name=domain_name,
                action_name=action_name,
            )
        except RuntimeError as e:
            retry_feedback = str(e)
            continue

        domain_text = _pddl_from_l2p_payload(
            domain_payload,
            "domain_pddl",
            "domain",
            "pddl",
        )
        if not domain_text:
            raise RuntimeError("Failed to generate domain PDDL from the request.")
        break
    else:
        raise RuntimeError(retry_feedback or "Failed to generate a valid domain.")

    return domain_name, domain_text


async def _run_problem_request(message: discord.Message, request_text: str) -> tuple[str, str]:
    retry_feedback: str | None = None
    domain_name = ""
    problem_name = ""
    problem_text = ""

    for _ in range(2):
        llm_plan = await _llm_plan_from_natural_language(
            message, request_text, retry_feedback
        )
        domain_name = str(llm_plan.get("domain_name", "")).strip()
        problem_name = str(llm_plan.get("problem_name", "")).strip()
        task_update = str(llm_plan.get("task_update", "")).strip()

        if not domain_name or not problem_name or not task_update:
            raise RuntimeError("The model did not return a complete problem update payload.")

        try:
            task_payload = await update_task_via_l2p(
                task_update=task_update,
                domain_name=domain_name,
                problem_name=problem_name,
            )
        except RuntimeError as e:
            retry_feedback = str(e)
            continue

        problem_text = _pddl_from_l2p_payload(
            task_payload,
            "task_pddl",
            "problem_pddl",
            "problem",
            "task",
            "pddl",
        )
        if not problem_text:
            raise RuntimeError("Failed to generate problem PDDL from the request.")
        break
    else:
        raise RuntimeError(retry_feedback or "Failed to generate a valid problem.")

    return problem_name, problem_text


async def _run_validate_plan_request(message: discord.Message) -> str:
    if message.attachments:
        domain_text, problem_text, plan_text = await _extract_val_attachments(message)
    else:
        cached = LAST_SOLVE_ARTIFACTS.get(_solve_artifacts_key(message))
        if not cached:
            return "Nothing to validate"
        domain_text = str(cached.get("domain", "")).strip()
        problem_text = str(cached.get("problem", "")).strip()
        plan_text = str(cached.get("plan", "")).strip()
        if not domain_text or not problem_text or not plan_text:
            return "Nothing to validate"
    result = await validate_plan(domain_text, problem_text, plan_text)
    return _format_validation_result("plan", result or "Plan validation returned an empty response.")


async def _run_validate_domain_request(message: discord.Message) -> str:
    if message.attachments:
        domain_text = await _extract_domain_attachment(message)
    else:
        cached = LAST_SOLVE_ARTIFACTS.get(_solve_artifacts_key(message))
        if not cached:
            return "Nothing to validate"
        domain_text = str(cached.get("domain", "")).strip()
        if not domain_text:
            return "Nothing to validate"
    result = await validate_domain(domain_text)
    return _format_validation_result("domain", result or "Domain validation returned an empty response.")


async def _run_validate_task_request(message: discord.Message) -> str:
    if message.attachments:
        domain_text, problem_text = await _extract_pddl_attachments(message)
    else:
        cached = LAST_SOLVE_ARTIFACTS.get(_solve_artifacts_key(message))
        if not cached:
            return "Nothing to validate"
        domain_text = str(cached.get("domain", "")).strip()
        problem_text = str(cached.get("problem", "")).strip()
        if not domain_text or not problem_text:
            return "Nothing to validate"
    result = await validate_task(domain_text, problem_text)
    return _format_validation_result("task", result or "Task validation returned an empty response.")


async def _run_autovalidate_request(
    message: discord.Message,
    max_iterations: int = 3,
) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    domain_text = _artifact_text(current, "domain")
    problem_text = _artifact_text(current, "problem")
    plan_text = _artifact_text(current, "plan")
    domain_name = str(current.get("domain_name", "domain")).strip() or "domain"
    problem_name = str(current.get("problem_name", "problem")).strip() or "problem"

    if not domain_text or not problem_text or not plan_text:
        raise RuntimeError("No current domain, problem, and plan to validate. Run `!plan` first.")

    working_domain = domain_text
    working_problem = problem_text
    val_text = ""

    for iteration in range(1, max_iterations + 1):
        val_text = await validate_plan_with_val(working_domain, working_problem, plan_text)
        if _val_output_indicates_valid(val_text):
            updated = _update_working_artifacts(
                message,
                domain=working_domain,
                problem=working_problem,
                plan=plan_text,
                domain_name=domain_name,
                problem_name=problem_name,
            )
            files = [
                _text_file(working_domain, _artifact_filename(updated, "domain")),
                _text_file(working_problem, _artifact_filename(updated, "problem")),
                _text_file(plan_text, _artifact_filename(updated, "plan")),
                _text_file(val_text, "val.log"),
            ]
            return (
                f"VAL accepted the plan after {iteration} iteration(s).",
                files,
            )

        feedback = (
            "Revise the current domain and problem so the existing plan remains unchanged and becomes valid under VAL. "
            "Make the smallest possible changes, preserve the user's original intent, and do not rewrite unrelated parts.\n"
            f"VAL feedback:\n{val_text}"
        )
        repair_feedback = val_text
        repaired = False
        for _ in range(2):
            domain_edit = await _llm_domain_edit_from_instruction(
                message,
                feedback,
                working_domain,
                domain_name,
                repair_feedback,
            )
            problem_edit = await _llm_problem_edit_from_instruction(
                message,
                feedback,
                working_problem,
                domain_name,
                problem_name,
                repair_feedback,
            )

            try:
                domain_payload = await _update_domain_with_fallback(
                    domain_update=str(domain_edit.get("domain_update", "")).strip(),
                    domain_name=str(domain_edit.get("domain_name", domain_name)).strip() or domain_name,
                    action_name=domain_edit.get("action_name"),
                )
                task_payload = await update_task_via_l2p(
                    task_update=str(problem_edit.get("task_update", "")).strip(),
                    domain_name=str(problem_edit.get("domain_name", domain_name)).strip() or domain_name,
                    problem_name=str(problem_edit.get("problem_name", problem_name)).strip() or problem_name,
                )
            except RuntimeError as e:
                repair_feedback = str(e)
                feedback = (
                    "Revise the current domain and problem so the existing plan remains unchanged and becomes valid under VAL. "
                    "Make the smallest possible changes, preserve the user's original intent, and do not rewrite unrelated parts. "
                    "The previous repair attempt also failed inside the L2P tools, so return a more complete and server-safe update.\n"
                    f"VAL feedback:\n{val_text}\n\nL2P feedback:\n{repair_feedback}"
                )
                continue

            domain_name = str(domain_edit.get("domain_name", domain_name)).strip() or domain_name
            problem_name = str(problem_edit.get("problem_name", problem_name)).strip() or problem_name
            working_domain = _pddl_from_l2p_payload(domain_payload, "domain_pddl", "domain", "pddl")
            working_problem = _pddl_from_l2p_payload(
                task_payload, "task_pddl", "problem_pddl", "problem", "task", "pddl"
            )
            if not working_domain or not working_problem:
                raise RuntimeError("Repair loop failed to produce revised domain/problem PDDL.")
            repaired = True
            break

        if not repaired:
            raise RuntimeError(
                f"Repair loop could not produce a valid domain/problem update.\n\nLast tool feedback:\n{repair_feedback}"
            )

    raise RuntimeError(
        f"VAL still rejected the plan after {max_iterations} iteration(s).\n\nLast VAL output:\n{val_text}"
    )


def _normalize_artifact_type(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"domain", "problem", "plan"}:
        return value
    return None


def _detect_artifact_request(text: str) -> tuple[str, str, str] | None:
    normalized = " ".join((text or "").strip().split())
    lowered = normalized.lower()
    show_match = re.match(r"^(show|display)\s+(the\s+)?(domain|problem|plan)\b", lowered)
    if show_match:
        artifact_type = show_match.group(3)
        return ("show", artifact_type, "")

    undo_match = re.match(r"^(undo|revert|restore)\s+(the\s+)?(domain|problem|plan)\b", lowered)
    if undo_match:
        artifact_type = undo_match.group(3)
        return ("undo", artifact_type, "")

    edit_match = re.match(
        r"^(edit|change|update|fix|revise)\s+(the\s+)?(domain|problem|plan)\b[:\s,-]*(.*)$",
        normalized,
        re.IGNORECASE,
    )
    if edit_match:
        artifact_type = edit_match.group(3).lower()
        instruction = edit_match.group(4).strip()
        return ("edit", artifact_type, instruction)

    for artifact_type in ("domain", "problem", "plan"):
        if artifact_type not in lowered:
            continue
        if any(
            marker in lowered
            for marker in ("don't like", "do not like", "should", "needs to", "must", "wrong")
        ):
            return ("edit", artifact_type, normalized)
    return None


async def _run_show_artifact_request(
    message: discord.Message, artifact_type: str
) -> tuple[str, list[discord.File] | None]:
    current = _working_artifacts(message)
    text = _artifact_text(current, artifact_type)
    if not text:
        raise RuntimeError(f"No current {artifact_type} to show.")
    reply_text = _artifact_reply_text(current, artifact_type)
    return reply_text, None


async def _run_explain_artifact_request(message: discord.Message, artifact_type: str) -> str:
    current = _working_artifacts(message)
    text = _artifact_text(current, artifact_type)
    if not text:
        raise RuntimeError(f"No current {artifact_type} to explain.")
    explanation = await _llm_explain_artifact(message, artifact_type, text)
    if not explanation:
        raise RuntimeError(f"Failed to explain the current {artifact_type}.")
    header = f"{artifact_type.capitalize()} explanation:"
    return _truncate_discord_message(f"{header}\n{explanation}")


async def _run_files_request(message: discord.Message) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    domain_text = _artifact_text(current, "domain")
    problem_text = _artifact_text(current, "problem")
    plan_text = _artifact_text(current, "plan")

    if not domain_text and not problem_text and not plan_text:
        raise RuntimeError("No current domain, problem, or plan to send.")

    files: list[discord.File] = []
    if domain_text:
        files.append(_text_file(domain_text, _artifact_filename(current, "domain")))
    if problem_text:
        files.append(_text_file(problem_text, _artifact_filename(current, "problem")))
    if plan_text:
        files.append(_text_file(plan_text, _artifact_filename(current, "plan")))

    return "Current artifact files.", files


async def _run_edit_domain_request(message: discord.Message, instruction: str) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    current_domain = _artifact_text(current, "domain")
    if not current_domain:
        raise RuntimeError("No current domain to edit. Generate or plan something first.")

    retry_feedback: str | None = None
    domain_name = str(current.get("domain_name", "domain")).strip() or "domain"
    for _ in range(2):
        edit = await _llm_domain_pddl_edit_from_instruction(
            message,
            instruction,
            current_domain,
            domain_name,
            retry_feedback,
        )
        domain_name = str(edit.get("domain_name", domain_name)).strip() or domain_name
        domain_text = str(edit.get("domain_pddl", "")).strip()
        if not domain_text:
            raise RuntimeError("Failed to generate revised domain PDDL.")
        validation_result = await validate_domain(domain_text)
        if not _validation_indicates_valid("domain", validation_result):
            retry_feedback = _format_validation_result("domain", validation_result)
            continue
        current = _update_working_artifacts(message, domain=domain_text, domain_name=domain_name)
        return (
            _format_updated_pddl_reply("domain", domain_name, domain_text),
            None,
        )
    raise RuntimeError(retry_feedback or "Failed to revise the domain.")


async def _run_edit_problem_request(message: discord.Message, instruction: str) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    current_domain = _artifact_text(current, "domain")
    current_problem = _artifact_text(current, "problem")
    if not current_domain:
        raise RuntimeError("No current domain for the problem to reference. Generate or plan something first.")
    if not current_problem:
        raise RuntimeError("No current problem to edit. Generate or plan something first.")

    retry_feedback: str | None = None
    domain_name = str(current.get("domain_name", "domain")).strip() or "domain"
    problem_name = str(current.get("problem_name", "problem")).strip() or "problem"
    for _ in range(2):
        edit = await _llm_problem_pddl_edit_from_instruction(
            message,
            instruction,
            current_domain,
            current_problem,
            domain_name,
            problem_name,
            retry_feedback,
        )
        domain_name = str(edit.get("domain_name", domain_name)).strip() or domain_name
        problem_name = str(edit.get("problem_name", problem_name)).strip() or problem_name
        problem_text = str(edit.get("problem_pddl", "")).strip()
        if not problem_text:
            raise RuntimeError("Failed to generate revised problem PDDL.")
        validation_result = await validate_task(current_domain, problem_text)
        if not _validation_indicates_valid("task", validation_result):
            retry_feedback = _format_validation_result("task", validation_result)
            continue
        current = _update_working_artifacts(
            message,
            problem=problem_text,
            problem_name=problem_name,
            domain_name=domain_name,
        )
        return (
            _format_updated_pddl_reply("problem", problem_name, problem_text),
            None,
        )
    raise RuntimeError(retry_feedback or "Failed to revise the problem.")


async def _run_edit_plan_request(message: discord.Message, instruction: str) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    current_plan = _artifact_text(current, "plan")
    if not current_plan:
        raise RuntimeError("No current plan to edit. Generate or plan something first.")

    revised_plan = await _llm_plan_edit_from_instruction(message, instruction, current_plan)
    current = _update_working_artifacts(message, plan=revised_plan)
    reply_text = f"Updated plan:\n```text\n{revised_plan}\n```"
    return reply_text, [_text_file(revised_plan, _artifact_filename(current, "plan"))]


async def _run_undo_request(message: discord.Message, artifact_type: str) -> tuple[str, list[discord.File] | None]:
    key = _solve_artifacts_key(message)
    history = ARTIFACT_HISTORY.get(key, [])
    if not history:
        raise RuntimeError("No previous artifact version to restore.")

    current = _working_artifacts(message)
    restored_index = None
    for index in range(len(history) - 1, -1, -1):
        snapshot = history[index]
        if _artifact_text(snapshot, artifact_type):
            restored_index = index
            current.update(snapshot)
            del history[index:]
            _persist_artifacts_if_session_saved()
            break
    if restored_index is None:
        raise RuntimeError(f"No previous {artifact_type} version to restore.")

    reply_text = _artifact_reply_text(current, artifact_type)
    files = [_text_file(_artifact_text(current, artifact_type), _artifact_filename(current, artifact_type))]
    return reply_text, files


async def _handle_workflow_request(message: discord.Message) -> bool:
    text = (message.content or "").strip()
    if not text:
        return False

    direct_request = _detect_artifact_request(text)
    if direct_request is not None:
        action, artifact_type, instruction = direct_request
        try:
            if action == "show":
                reply_text, files = await _run_show_artifact_request(message, artifact_type)
            elif action == "undo":
                reply_text, files = await _run_undo_request(message, artifact_type)
            elif action == "edit":
                if not instruction:
                    await message.reply(
                        f"Tell me how to revise the {artifact_type}.",
                        mention_author=False,
                    )
                    return True
                if artifact_type == "domain":
                    reply_text, files = await _run_edit_domain_request(message, instruction)
                elif artifact_type == "problem":
                    reply_text, files = await _run_edit_problem_request(message, instruction)
                else:
                    reply_text, files = await _run_edit_plan_request(message, instruction)
            else:
                return False
        except Exception as e:
            traceback.print_exc()
            await message.reply(
                _truncate_discord_message(f"HITL workflow failed: {type(e).__name__}: {e}"),
                mention_author=False,
            )
            return True

        messages = _split_discord_message(reply_text)
        await message.reply(messages[0], files=files or None, mention_author=False)
        for chunk in messages[1:]:
            await message.channel.send(chunk)
        return True

    try:
        intent = await _llm_classify_workflow_request(message)
    except Exception as e:
        print("[LLM WORKFLOW CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if intent == "chat":
        return False

    async with message.channel.typing():
        try:
            if intent == "help":
                reply_text = _format_help_message()
                files = None
            elif intent == "tools":
                reply_text = _format_mcp_tools_message(await list_all_mcp_tools())
                files = None
            elif intent == "plan":
                reply_text, files = await _run_plan_request(message, text)
            elif intent == "domain":
                domain_name, domain_text = await _run_domain_request(message, text)
                _update_working_artifacts(message, domain=domain_text, domain_name=domain_name)
                reply_text = _format_domain_reply(domain_name, domain_text)
                files = None
            elif intent == "problem":
                problem_name, problem_text = await _run_problem_request(message, text)
                _update_working_artifacts(
                    message,
                    problem=problem_text,
                    problem_name=problem_name,
                )
                reply_text = _format_problem_reply(problem_name, problem_text)
                files = None
            elif intent == "validate_plan":
                reply_text = await _run_validate_plan_request(message)
                files = None
            elif intent == "validate_domain":
                reply_text = await _run_validate_domain_request(message)
                files = None
            elif intent == "validate_task":
                reply_text = await _run_validate_task_request(message)
                files = None
            else:
                return False
        except Exception as e:
            traceback.print_exc()
            await message.reply(
                _truncate_discord_message(f"Workflow failed: {type(e).__name__}: {e}"),
                mention_author=False,
            )
            return True

    if files:
        await message.reply(_truncate_discord_message(reply_text), files=files, mention_author=False)
        return True

    messages = _split_discord_message(reply_text)
    await message.reply(messages[0], mention_author=False)
    for chunk in messages[1:]:
        await message.channel.send(chunk)
    return True


def _normalize_member_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def _member_name_variants(member: discord.Member) -> list[str]:
    variants = [member.display_name, member.name]
    global_name = getattr(member, "global_name", None)
    if global_name:
        variants.append(global_name)
    return [variant for variant in variants if variant]


def _score_member_match(query: str, member: discord.Member) -> float:
    normalized_query = _normalize_member_text(query)
    if not normalized_query:
        return 0.0

    best_score = 0.0
    query_tokens = set(normalized_query.split())

    for variant in _member_name_variants(member):
        normalized_variant = _normalize_member_text(variant)
        if not normalized_variant:
            continue

        ratio = difflib.SequenceMatcher(None, normalized_query, normalized_variant).ratio()
        token_overlap = len(query_tokens & set(normalized_variant.split()))
        token_bonus = 0.12 * token_overlap
        substring_bonus = (
            0.2
            if normalized_query in normalized_variant or normalized_variant in normalized_query
            else 0.0
        )
        prefix_bonus = 0.08 if normalized_variant.startswith(normalized_query) else 0.0

        best_score = max(best_score, ratio + token_bonus + substring_bonus + prefix_bonus)

    return best_score


async def _rank_matching_members(guild: discord.Guild, query: str) -> list[discord.Member]:
    exact_member = guild.get_member_named(query)
    if exact_member is not None:
        return [exact_member]

    members = list(guild.members)
    if not members:
        try:
            members = [member async for member in guild.fetch_members(limit=None)]
        except discord.HTTPException:
            return []

    scored_members: list[tuple[float, discord.Member]] = []
    for member in members:
        score = _score_member_match(query, member)
        if score >= 0.45:
            scored_members.append((score, member))

    scored_members.sort(
        key=lambda item: (
            -item[0],
            len(_normalize_member_text(item[1].display_name or item[1].name)),
        )
    )
    return [member for _, member in scored_members]


async def _handle_member_confirmation_response(message: discord.Message) -> bool:
    key = (message.channel.id, message.author.id)
    pending = PENDING_MEMBER_CONFIRMATIONS.get(key)
    if pending is None:
        return False

    try:
        reply_type = await _llm_classify_confirmation_reply(message)
    except Exception as e:
        print("[LLM CONFIRMATION CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if reply_type == "cancel":
        PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
        await message.reply("Okay, I stopped the member lookup.", mention_author=False)
        return True

    if reply_type == "reject":
        candidate_ids = pending["candidate_ids"]
        current_index = pending["current_index"] + 1
        guild = message.guild

        if guild is None:
            PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
            await message.reply("Okay, I won't mention them.", mention_author=False)
            return True

        next_member: discord.Member | None = None
        while current_index < len(candidate_ids):
            next_member = guild.get_member(candidate_ids[current_index])
            if next_member is not None:
                break
            current_index += 1

        if next_member is None:
            PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
            await message.reply(
                "I couldn't find another close match, so I'll stop here.",
                mention_author=False,
            )
            return True

        pending["current_index"] = current_index
        action = "mention them"
        if pending["should_add_to_thread"]:
            action = "mention them and add them to this thread"

        await message.reply(
            f"Okay, how about {next_member.mention}? Reply `yes` to {action}, or `no` if that's not the right user either.",
            mention_author=False,
        )
        return True

    if reply_type != "confirm":
        return False

    PENDING_MEMBER_CONFIRMATIONS.pop(key, None)

    guild = message.guild
    if guild is None:
        return True

    member = guild.get_member(pending["candidate_ids"][pending["current_index"]])
    if member is None:
        await message.reply("I couldn't find that member anymore.", mention_author=False)
        return True

    requested_name = pending["requested_name"]
    should_add_to_thread = pending["should_add_to_thread"]
    response = f"{member.mention}"

    if should_add_to_thread and isinstance(message.channel, discord.Thread):
        try:
            await message.channel.add_user(member)
            response = f"{member.mention} added to this thread."
        except discord.HTTPException:
            response = (
                f"{member.mention} is the closest match for `{requested_name}`, "
                "but I couldn't add them to this thread."
            )

    await message.reply(response, mention_author=False)
    return True


def _looks_like_solve_request(message: discord.Message) -> bool:
    if len(message.attachments) < 2:
        return False

    text = (message.content or "").lower()
    solve_words = ("solve", "plan", "planner", "pddl")

    if any(word in text for word in solve_words):
        return True

    filenames = " ".join(attachment.filename.lower() for attachment in message.attachments)
    return "domain" in filenames and "problem" in filenames


async def _handle_thread_add_request(message: discord.Message) -> bool:
    if not isinstance(message.channel, discord.Thread):
        return False

    if not message.mentions:
        return False

    try:
        decision = await _llm_classify_member_request(message)
    except Exception as e:
        print("[LLM MEMBER REQUEST CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if decision["intent"] != "thread_add_mentions":
        return False

    thread = message.channel
    added_users: list[str] = []
    failed_users: list[str] = []

    async with thread.typing():
        for user in message.mentions:
            if user.bot:
                failed_users.append(user.mention)
                continue

            try:
                await thread.add_user(user)
                added_users.append(user.mention)
            except discord.HTTPException:
                failed_users.append(user.mention)

    parts: list[str] = []
    if added_users:
        parts.append("Added to thread: " + ", ".join(added_users))
    if failed_users:
        parts.append("Couldn't add: " + ", ".join(failed_users))

    await message.reply(
        ". ".join(parts) or "I couldn't add anyone to this thread.",
        mention_author=False,
    )
    return True


async def _handle_member_lookup_request(message: discord.Message) -> bool:
    if message.guild is None:
        return False

    if message.mentions:
        return False

    try:
        decision = await _llm_classify_member_request(message)
    except Exception as e:
        print("[LLM MEMBER REQUEST CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if decision["intent"] != "member_lookup":
        return False

    requested_name = decision["requested_name"]
    if not requested_name:
        return False

    candidates = await _rank_matching_members(message.guild, requested_name)
    if not candidates:
        await message.reply(
            f"I couldn't find a close match for `{requested_name}` in this server.",
            mention_author=False,
        )
        return True

    member = candidates[0]
    should_add_to_thread = bool(decision["should_add_to_thread"]) and isinstance(
        message.channel, discord.Thread
    )
    PENDING_MEMBER_CONFIRMATIONS[(message.channel.id, message.author.id)] = {
        "candidate_ids": [candidate.id for candidate in candidates],
        "current_index": 0,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }

    action = "mention them"
    if should_add_to_thread:
        action = "mention them and add them to this thread"

    await message.reply(
        f"Did you mean {member.mention}? Reply `yes` to {action}, or `no` to cancel.",
        mention_author=False,
    )
    return True


async def _handle_solve_request(message: discord.Message) -> bool:
    if not _looks_like_solve_request(message):
        return False

    async with message.channel.typing():
        try:
            reply_text, files = await _run_plan_request(message, None)
        except Exception as e:
            traceback.print_exc()
            await message.reply(
                _truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}"),
                mention_author=False,
            )
            return True

    await message.reply(reply_text, files=files, mention_author=False)
    return True


async def _close_bot_with_mcp_cleanup():
    await close_mcp_servers()
    await commands.Bot.close(bot)


bot.close = _close_bot_with_mcp_cleanup


@bot.tree.command(name="models", description="List available models")
async def models_cmd(interaction: discord.Interaction):
    ids = _all_llm_model_ids()
    if not ids:
        await interaction.response.send_message(
            "No models found. Install an llm provider plugin.", ephemeral=True
        )
        return

    text = "Available models:\n" + "\n".join(f"- {mid}" for mid in ids)
    await interaction.response.send_message(text[:1900], ephemeral=True)


async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    ids = _all_llm_model_ids()
    current_lower = (current or "").lower()

    matches = [mid for mid in ids if current_lower in mid.lower()]
    matches = matches[:25]

    return [app_commands.Choice(name=mid, value=mid) for mid in matches]


@bot.tree.command(name="use", description="Choose the model this bot will use for you")
@app_commands.describe(model_id="Pick from /models")
@app_commands.autocomplete(model_id=model_autocomplete)
async def use_cmd(interaction: discord.Interaction, model_id: str):
    ids = set(_all_llm_model_ids())
    if model_id not in ids:
        await interaction.response.send_message(
            "Unknown model_id. Use /models.", ephemeral=True
        )
        return

    set_user_model(str(interaction.user.id), model_id)
    await interaction.response.send_message(f"Now using: {model_id}", ephemeral=True)


@bot.tree.command(name="setkey", description="Set your API key for a provider")
@app_commands.describe(provider="openai | gemini | anthropic", api_key="Your API key")
async def setkey_cmd(interaction: discord.Interaction, provider: str, api_key: str):
    provider = provider.strip().lower()
    if provider not in ("openai", "gemini", "anthropic"):
        await interaction.response.send_message(
            "Unknown provider. Use: openai | gemini | anthropic",
            ephemeral=True,
        )
        return

    save_provider_key(
        user_id=str(interaction.user.id),
        provider=provider,
        api_key=api_key.strip(),
    )
    await interaction.response.send_message(
        f"Saved key for {provider}.",
        ephemeral=True,
    )


@bot.event
async def on_ready():
    await connect_mcp_servers()
    saved_artifacts, saved_history = load_saved_working_artifacts()
    LAST_SOLVE_ARTIFACTS.clear()
    LAST_SOLVE_ARTIFACTS.update(saved_artifacts)
    ARTIFACT_HISTORY.clear()
    ARTIFACT_HISTORY.update(saved_history)
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    guild = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild)
    synced = await bot.tree.sync(guild=guild)
    print(f"Synced {len(synced)} commands to guild {GUILD_ID}.")


async def _handle_conversation_message(message: discord.Message):
    if await _handle_member_confirmation_response(message):
        return

    if await _handle_thread_add_request(message):
        return

    if await _handle_member_lookup_request(message):
        return

    if await _handle_solve_request(message):
        return

    if await _handle_workflow_request(message):
        return

    text = (message.content or "").strip()
    if not text:
        return

    log_message(
        message.channel.id,
        message.author.id,
        "user",
        text,
        message.guild.id if message.guild else None,
    )

    async with message.channel.typing():
        context = get_recent_context(
            user_id=message.author.id,
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
        )
        selected_model = get_user_model(str(message.author.id)) or MODEL
        try:
            answer = await llm_reply(
                selected_model, context, user_id=str(message.author.id)
            )
        except Exception as e:
            print("[LLM ERROR]", type(e).__name__, e)
            traceback.print_exc()
            answer = f"Error generating response: {type(e).__name__}: {e}"

    log_message(
        message.channel.id,
        message.author.id,
        "assistant",
        answer,
        message.guild.id if message.guild else None,
    )

    await message.reply(_truncate_discord_message(answer), mention_author=False)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return

    if not is_chat_enabled(str(message.author.id), str(message.channel.id)):
        return

    await _handle_conversation_message(message)


@bot.command()
async def thread(ctx: commands.Context, *, topic: str | None = None):
    if ctx.guild is None:
        await ctx.reply("Thread creation only works in a server channel.")
        return

    if isinstance(ctx.channel, discord.Thread):
        await ctx.reply("This is already a thread.")
        return

    if not isinstance(ctx.channel, discord.TextChannel):
        await ctx.reply("Create a chat thread from a regular server text channel.")
        return

    thread_name = (topic or f"Chat with {ctx.author.display_name}").strip()
    thread_name = thread_name[:100] or f"Chat with {ctx.author.display_name}"

    try:
        thread = await ctx.channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
            invitable=False,
        )
    except discord.HTTPException:
        await ctx.reply("I couldn't create a thread here. Check my thread permissions.")
        return

    try:
        await thread.add_user(ctx.author)
    except discord.HTTPException:
        pass

    await ctx.reply(f"Started a chat thread: {thread.mention}")


@bot.command()
async def help(ctx: commands.Context):
    messages = _split_discord_message(_format_help_message())
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def chat(ctx: commands.Context):
    user_id = str(ctx.author.id)
    channel_id = str(ctx.channel.id)
    enabled = not is_chat_enabled(user_id, channel_id)
    set_chat_enabled(user_id, channel_id, enabled)

    if enabled:
        await ctx.reply("LLM chat is now `on` in this channel for you.")
        return

    await ctx.reply(
        "LLM chat is now `off` in this channel for you. Your normal messages here will no longer call the bot until you run `!chat` again."
    )


@bot.command(name="plan")
async def plan_cmd(ctx: commands.Context, *, request: str | None = None):
    async with ctx.typing():
        try:
            reply_text, files = await _run_plan_request(ctx.message, request)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(reply_text, files=files)


@bot.command(name="domain")
async def domain_cmd(ctx: commands.Context, *, request: str | None = None):
    request_text = (request or "").strip()
    if not request_text:
        await ctx.reply("Use `!domain <natural language request>`.")
        return

    async with ctx.typing():
        try:
            domain_name, domain_text = await _run_domain_request(ctx.message, request_text)
            _update_working_artifacts(ctx.message, domain=domain_text, domain_name=domain_name)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Domain generation failed: {type(e).__name__}: {e}")
            )
            return
    reply_text = _format_domain_reply(domain_name, domain_text)
    messages = _split_discord_message(reply_text)
    await ctx.reply(
        messages[0],
        file=_text_file(domain_text, f"{_safe_pddl_name(domain_name, 'domain')}.pddl"),
    )
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="problem")
async def problem_cmd(ctx: commands.Context, *, request: str | None = None):
    request_text = (request or "").strip()
    if not request_text:
        await ctx.reply("Use `!problem <natural language request>`.")
        return

    async with ctx.typing():
        try:
            problem_name, problem_text = await _run_problem_request(ctx.message, request_text)
            _update_working_artifacts(
                ctx.message, problem=problem_text, problem_name=problem_name
            )
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Problem generation failed: {type(e).__name__}: {e}")
            )
            return
    reply_text = _format_problem_reply(problem_name, problem_text)
    messages = _split_discord_message(reply_text)
    await ctx.reply(
        messages[0],
        file=_text_file(problem_text, f"{_safe_pddl_name(problem_name, 'problem')}.pddl"),
    )
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="show")
async def show_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!show domain`, `!show problem`, or `!show plan`.")
        return

    async with ctx.typing():
        try:
            reply_text, files = await _run_show_artifact_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Show failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="files")
async def files_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            reply_text, files = await _run_files_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Files failed: {type(e).__name__}: {e}")
            )
            return

    await ctx.reply(reply_text, files=files)


@bot.command(name="explain")
async def explain_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!explain domain`, `!explain problem`, or `!explain plan`.")
        return

    async with ctx.typing():
        try:
            reply_text = await _run_explain_artifact_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Explain failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="edit")
async def edit_cmd(ctx: commands.Context, artifact_type: str, *, instruction: str | None = None):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!edit domain ...`, `!edit problem ...`, or `!edit plan ...`.")
        return

    edit_instruction = (instruction or "").strip()
    if not edit_instruction:
        await ctx.reply(f"Tell me how to revise the {normalized}.")
        return

    async with ctx.typing():
        try:
            if normalized == "domain":
                reply_text, files = await _run_edit_domain_request(ctx.message, edit_instruction)
            elif normalized == "problem":
                reply_text, files = await _run_edit_problem_request(ctx.message, edit_instruction)
            else:
                reply_text, files = await _run_edit_plan_request(ctx.message, edit_instruction)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Edit failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="undo")
async def undo_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!undo domain`, `!undo problem`, or `!undo plan`.")
        return

    async with ctx.typing():
        try:
            reply_text, files = await _run_undo_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Undo failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="validate")
async def validate_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await _run_validate_plan_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Plan validation failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(result)


@bot.command(name="validate_plan")
async def validate_plan_cmd(ctx: commands.Context):
    await validate_cmd(ctx)


@bot.command(name="validate_domain")
async def validate_domain_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await _run_validate_domain_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(
                    f"Domain validation failed: {type(e).__name__}: {e}"
                )
            )
            return

    await ctx.reply(result)


@bot.command(name="validate_task")
async def validate_task_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await _run_validate_task_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(
                    f"Task validation failed: {type(e).__name__}: {e}"
                )
            )
            return

    await ctx.reply(result)


@bot.command(name="autovalidate")
async def autovalidate_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            reply_text, files = await _run_autovalidate_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Auto-validate failed: {type(e).__name__}: {e}")
            )
            return

    await ctx.reply(reply_text, files=files)


@bot.command(name="paastools")
async def paastools_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            tool_catalog = await get_mcp_tool_catalog()
            paas_tools = tool_catalog.get("paas", [])
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(
                    f"PaaS tool listing failed: {type(e).__name__}: {e}"
                )
            )
            return

    messages = _split_discord_message(
        _format_single_server_tools_message("paas", paas_tools)
    )
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def tools(ctx: commands.Context):
    async with ctx.typing():
        try:
            tool_map = await list_all_mcp_tools()
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(
                    f"Tool listing failed: {type(e).__name__}: {e}"
                )
            )
            return

    messages = _split_discord_message(_format_mcp_tools_message(tool_map))
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def share(ctx: commands.Context):
    user_id = str(ctx.author.id)

    if not user_has_any_provider_key(user_id):
        await ctx.reply("Set an API key first with `/setkey`, then you can toggle sharing.")
        return

    current_mode = get_share_mode(user_id)
    new_mode = "group" if current_mode == "individual" else "individual"
    set_share_mode(user_id, new_mode)

    if new_mode == "group":
        await ctx.reply(
            "Share mode is now `group`. Other users can use the bot through your saved API key when they do not have their own key for that provider."
        )
        return

    await ctx.reply(
        "Share mode is now `individual`. Only you can use your saved API key."
    )


@bot.command()
async def save(ctx: commands.Context):
    is_new_save = save_current_session()
    save_working_artifacts_snapshot(
        BOT_SESSION_ID,
        LAST_SOLVE_ARTIFACTS,
        ARTIFACT_HISTORY,
    )
    if is_new_save:
        await ctx.reply("Saved the current bot session. Its conversation log will be kept after restart.")
        return

    await ctx.reply("This bot session was already saved.")

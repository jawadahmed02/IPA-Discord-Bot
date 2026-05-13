import traceback

import discord

from IPA_Discbot.mcp_client import (
    list_all_mcp_tools,
    parse_solve_response_text as _parse_solve_response_text,
    solve_pddl,
    update_domain_via_l2p,
    update_task_via_l2p,
    validate_domain,
    validate_plan,
    validate_plan_with_val,
    validate_task,
)
from .llm_helpers import (
    _llm_classify_confirmation_reply,
    _llm_classify_member_request,
    _llm_classify_workflow_request,
    _llm_domain_edit_from_instruction,
    _llm_domain_pddl_edit_from_instruction,
    _llm_explain_artifact,
    _llm_plan_edit_from_instruction,
    _llm_plan_from_natural_language,
    _llm_problem_edit_from_instruction,
    _llm_problem_pddl_edit_from_instruction,
    llm_reply,
)
from .config import LAST_SOLVE_ARTIFACTS, MODEL, PENDING_MEMBER_CONFIRMATIONS
from .parsing import (
    _collect_validation_text,
    _detect_artifact_request,
    _extract_domain_attachment,
    _extract_pddl_attachments,
    _extract_val_attachments,
    _extract_validation_payload,
    _pddl_from_l2p_payload,
    _planner_output_indicates_failure,
    _rank_matching_members,
    _solve_output_has_action_steps,
    _split_discord_message,
    _summarize_validation_failure,
    _truncate_discord_message,
    _validation_indicates_valid,
    _val_output_indicates_valid,
)
from .state import (
    _artifact_text,
    _restore_artifact_version,
    _solve_artifacts_key,
    _store_last_solve_artifacts,
    _update_working_artifacts,
    _working_artifacts,
)
from .storage import (
    get_recent_context,
    get_user_model,
    is_collab_enabled,
    log_message,
)


# ---------------------------------------------------------------------------
# Formatting helpers (used by both workflows and command handlers)
# ---------------------------------------------------------------------------

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


def _format_single_server_tools_message(server_name: str, tools: list[dict[str, str]]) -> str:
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
        "`!help` Show this help message with a short description of each command.",
        "",
        "Planning:",
        "`!plan <request>` Solve a planning request from natural language, or attach domain/problem PDDL files with `!plan`.",
        "`!plan` Re-solve using the current stored domain and problem in this channel for you.",
        "`!domain <request>` Generate a planning domain from a natural-language request and return the domain PDDL.",
        "`!problem <request>` Generate a planning problem from a natural-language request and return the problem PDDL.",
        "",
        "HITL Editing:",
        "`!files` Send the current stored domain, problem, and plan as files.",
        "`!show <domain|problem|plan>` Show the current working artifact in chat.",
        "`!explain <domain|problem|plan>` Explain the current working artifact in natural language.",
        "`!edit <domain|problem|plan> <instruction>` Revise one current artifact while preserving the rest of the workflow state.",
        "`!undo <domain|problem>` Restore the previous version of one user-authored artifact.",
        "",
        "Validation:",
        "`!validate` Validate that a plan fits a domain and problem with VAL, using attachments or the last successful `!plan` output.",
        "`!validate_domain` Validate a domain PDDL file with the PaaS domain validation tool.",
        "`!validate_task` Validate a domain/problem pair with the PaaS task validation tool.",
        "`!validate_plan` Validate a domain/problem/plan triple with the PaaS plan validation tool.",
        "",
        "Progress / Beta:",
        "`!autovalidate` Beta: loop domain/problem repairs against VAL until the current plan passes or the retry limit is reached.",
        "",
        "MCP Tools:",
        "`!tools` List the MCP tools currently exposed by the connected planning servers.",
        "`!paastools` List only the tools currently exposed by the connected PaaS MCP server.",
        "",
        "Chat and Threads:",
        "`!thread [topic]` Create a private thread in the current server channel for chatting with the bot.",
        "`!chat` Toggle normal-message bot chat on or off for you in the current channel.",
        "`!collab` Toggle shared collaboration mode for this channel or thread.",
        "",
        "Bot Settings:",
        "`!share` Toggle whether other users may use your saved provider key when they do not have one.",
        "`!save` Save the current bot session so its conversation log is kept after restart.",
        "`/models` Show the available LLM model IDs you can choose from.",
        "`/use <model_id>` Set which model the bot should use for your requests.",
        "`/setkey <provider> <api_key>` Save your provider API key for `openai`, `gemini`, or `anthropic`.",
    ]
    return "\n".join(lines)


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


def _format_validation_result(kind: str, raw: object) -> str:
    title, _ = ("Domain", "domain") if kind == "domain" else (
        ("Task", "task") if kind == "task" else ("Plan", "plan")
    )
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


def _artifact_reply_text(current: dict[str, str], artifact_type: str) -> str:
    text = _artifact_text(current, artifact_type)
    if not text:
        return ""
    if artifact_type == "domain":
        return _format_domain_reply(str(current.get("domain_name", "")).strip(), text)
    if artifact_type == "problem":
        return _format_problem_reply(str(current.get("problem_name", "")).strip(), text)
    return f"Current plan:\n```text\n{text}\n```"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        if "More action_name values were provided than action updates" not in str(e):
            raise
        return await update_domain_via_l2p(
            domain_update=domain_update,
            domain_name=domain_name,
        )


async def _extract_current_plan_validation_inputs(
    message: discord.Message,
) -> tuple[str, str, str] | None:
    if message.attachments:
        return await _extract_val_attachments(message)

    cached = LAST_SOLVE_ARTIFACTS.get(_solve_artifacts_key(message))
    if not cached:
        return None

    domain_text = str(cached.get("domain", "")).strip()
    problem_text = str(cached.get("problem", "")).strip()
    plan_text = str(cached.get("plan", "")).strip()
    if not domain_text or not problem_text or not plan_text:
        return None

    return domain_text, problem_text, plan_text


def _looks_like_solve_request(message: discord.Message) -> bool:
    if len(message.attachments) < 2:
        return False

    text = (message.content or "").lower()
    if any(word in text for word in ("solve", "plan", "planner", "pddl")):
        return True

    filenames = " ".join(a.filename.lower() for a in message.attachments)
    return "domain" in filenames and "problem" in filenames


def _shared_log_content(message: discord.Message) -> str:
    display_name = message.author.display_name.strip() or message.author.name
    return f"{display_name}: {(message.content or '').strip()}"


# ---------------------------------------------------------------------------
# Planning runners
# ---------------------------------------------------------------------------

async def run_plan_request(
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
            if _planner_output_indicates_failure(result) and not _solve_output_has_action_steps(result):
                raise RuntimeError(result or "Failed to produce a valid plan from the current domain and problem.")
        else:
            result = ""
            retry_feedback: str | None = None

            for _ in range(2):
                llm_plan = await _llm_plan_from_natural_language(message, request_text, retry_feedback)
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

                domain_text = _pddl_from_l2p_payload(domain_payload, "domain_pddl", "domain", "pddl")
                problem_text = _pddl_from_l2p_payload(
                    task_payload, "task_pddl", "problem_pddl", "problem", "task", "pddl"
                )

                if not domain_text or not problem_text:
                    raise RuntimeError("Failed to generate PDDL from the natural-language request.")

                raw_result = await solve_pddl(domain_text, problem_text)
                result = _parse_solve_response_text(raw_result)
                if _planner_output_indicates_failure(result) and not _solve_output_has_action_steps(result):
                    retry_feedback = result
                    continue
                break
            else:
                raise RuntimeError(retry_feedback or "Failed to produce a valid plan.")

    _store_last_solve_artifacts(message, domain_text, problem_text, result, domain_name, problem_name)
    return f"Generated planning artifacts.\n\nCurrent plan:\n```text\n{result.strip()}\n```", None


async def run_domain_request(message: discord.Message, request_text: str) -> tuple[str, str]:
    retry_feedback: str | None = None
    domain_name = ""
    domain_text = ""

    for _ in range(2):
        llm_plan = await _llm_plan_from_natural_language(message, request_text, retry_feedback)
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

        domain_text = _pddl_from_l2p_payload(domain_payload, "domain_pddl", "domain", "pddl")
        if not domain_text:
            raise RuntimeError("Failed to generate domain PDDL from the request.")
        break
    else:
        raise RuntimeError(retry_feedback or "Failed to generate a valid domain.")

    return domain_name, domain_text


async def run_problem_request(message: discord.Message, request_text: str) -> tuple[str, str]:
    retry_feedback: str | None = None
    domain_name = ""
    problem_name = ""
    problem_text = ""

    for _ in range(2):
        llm_plan = await _llm_plan_from_natural_language(message, request_text, retry_feedback)
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
            task_payload, "task_pddl", "problem_pddl", "problem", "task", "pddl"
        )
        if not problem_text:
            raise RuntimeError("Failed to generate problem PDDL from the request.")
        break
    else:
        raise RuntimeError(retry_feedback or "Failed to generate a valid problem.")

    return problem_name, problem_text


async def run_validate_request(message: discord.Message) -> str:
    inputs = await _extract_current_plan_validation_inputs(message)
    if inputs is None:
        return "Nothing to validate"
    domain_text, problem_text, plan_text = inputs
    result = await validate_plan_with_val(domain_text, problem_text, plan_text)
    return _format_validation_result("plan", result or "VAL validation returned an empty response.")


async def run_validate_plan_request(message: discord.Message) -> str:
    inputs = await _extract_current_plan_validation_inputs(message)
    if inputs is None:
        return "Nothing to validate"
    domain_text, problem_text, plan_text = inputs
    result = await validate_plan(domain_text, problem_text, plan_text)
    return _format_validation_result("plan", result or "Plan validation returned an empty response.")


async def run_validate_domain_request(message: discord.Message) -> str:
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


async def run_validate_task_request(message: discord.Message) -> str:
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


async def run_autovalidate_request(
    message: discord.Message,
    max_iterations: int = 3,
) -> tuple[str, list[discord.File] | None]:
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
            _update_working_artifacts(
                message,
                domain=working_domain,
                problem=working_problem,
                plan=plan_text,
                domain_name=domain_name,
                problem_name=problem_name,
            )
            return f"VAL accepted the plan after {iteration} iteration(s).", None

        feedback = (
            "Revise the current domain and problem so the existing plan remains unchanged and becomes valid under VAL. "
            "Make the smallest possible changes, preserve the user's original intent, and do not rewrite unrelated parts.\n"
            f"VAL feedback:\n{val_text}"
        )
        repair_feedback = val_text
        repaired = False
        for _ in range(2):
            domain_edit = await _llm_domain_edit_from_instruction(
                message, feedback, working_domain, domain_name, repair_feedback
            )
            problem_edit = await _llm_problem_edit_from_instruction(
                message, feedback, working_problem, domain_name, problem_name, repair_feedback
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


async def run_show_artifact_request(
    message: discord.Message, artifact_type: str
) -> tuple[str, list[discord.File] | None]:
    current = _working_artifacts(message)
    if not _artifact_text(current, artifact_type):
        raise RuntimeError(f"No current {artifact_type} to show.")
    return _artifact_reply_text(current, artifact_type), None


async def run_explain_artifact_request(message: discord.Message, artifact_type: str) -> str:
    current = _working_artifacts(message)
    text = _artifact_text(current, artifact_type)
    if not text:
        raise RuntimeError(f"No current {artifact_type} to explain.")
    explanation = await _llm_explain_artifact(message, artifact_type, text)
    if not explanation:
        raise RuntimeError(f"Failed to explain the current {artifact_type}.")
    return _truncate_discord_message(f"{artifact_type.capitalize()} explanation:\n{explanation}")


async def run_files_request(message: discord.Message) -> tuple[str, list[discord.File]]:
    import io
    import re

    current = _working_artifacts(message)
    domain_text = _artifact_text(current, "domain")
    problem_text = _artifact_text(current, "problem")
    plan_text = _artifact_text(current, "plan")

    if not domain_text and not problem_text and not plan_text:
        raise RuntimeError("No current domain, problem, or plan to send.")

    def _text_file(text: str, filename: str) -> discord.File:
        return discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)

    def _safe_pddl_name(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_")
        return cleaned or fallback

    def _artifact_filename(artifact_type: str) -> str:
        if artifact_type == "domain":
            return f"{_safe_pddl_name(current.get('domain_name', ''), 'domain')}.pddl"
        if artifact_type == "problem":
            return f"{_safe_pddl_name(current.get('problem_name', ''), 'problem')}.pddl"
        return "plan.txt"

    files: list[discord.File] = []
    if domain_text:
        files.append(_text_file(domain_text, _artifact_filename("domain")))
    if problem_text:
        files.append(_text_file(problem_text, _artifact_filename("problem")))
    if plan_text:
        files.append(_text_file(plan_text, _artifact_filename("plan")))

    return "Current artifact files.", files


async def run_edit_domain_request(message: discord.Message, instruction: str) -> tuple[str, list[discord.File]]:
    current = _working_artifacts(message)
    current_domain = _artifact_text(current, "domain")
    if not current_domain:
        raise RuntimeError("No current domain to edit. Generate or plan something first.")

    retry_feedback: str | None = None
    domain_name = str(current.get("domain_name", "domain")).strip() or "domain"
    for _ in range(2):
        edit = await _llm_domain_pddl_edit_from_instruction(
            message, instruction, current_domain, domain_name, retry_feedback
        )
        domain_name = str(edit.get("domain_name", domain_name)).strip() or domain_name
        domain_text = str(edit.get("domain_pddl", "")).strip()
        if not domain_text:
            raise RuntimeError("Failed to generate revised domain PDDL.")
        validation_result = await validate_domain(domain_text)
        if not _validation_indicates_valid("domain", validation_result):
            retry_feedback = _format_validation_result("domain", validation_result)
            continue
        _update_working_artifacts(message, domain=domain_text, domain_name=domain_name)
        return _format_updated_pddl_reply("domain", domain_name, domain_text), None
    raise RuntimeError(retry_feedback or "Failed to revise the domain.")


async def run_edit_problem_request(message: discord.Message, instruction: str) -> tuple[str, list[discord.File]]:
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
            message, instruction, current_domain, current_problem, domain_name, problem_name, retry_feedback
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
        _update_working_artifacts(message, problem=problem_text, problem_name=problem_name, domain_name=domain_name)
        return _format_updated_pddl_reply("problem", problem_name, problem_text), None
    raise RuntimeError(retry_feedback or "Failed to revise the problem.")


async def run_edit_plan_request(
    message: discord.Message, instruction: str
) -> tuple[str, list[discord.File] | None]:
    current = _working_artifacts(message)
    current_plan = _artifact_text(current, "plan")
    if not current_plan:
        raise RuntimeError("No current plan to edit. Generate or plan something first.")
    revised_plan = await _llm_plan_edit_from_instruction(message, instruction, current_plan)
    _update_working_artifacts(message, plan=revised_plan)
    return f"Updated plan:\n```text\n{revised_plan}\n```", None


async def run_undo_request(
    message: discord.Message, artifact_type: str
) -> tuple[str, list[discord.File] | None]:
    current = _restore_artifact_version(message, artifact_type)
    return _artifact_reply_text(current, artifact_type), None


# ---------------------------------------------------------------------------
# Message-level handlers (routing logic called from on_message)
# ---------------------------------------------------------------------------

async def handle_member_confirmation_response(message: discord.Message) -> bool:
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
                "I couldn't find another close match, so I'll stop here.", mention_author=False
            )
            return True

        pending["current_index"] = current_index
        action = "mention them and add them to this thread" if pending["should_add_to_thread"] else "mention them"
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


async def handle_thread_add_request(message: discord.Message) -> bool:
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
        ". ".join(parts) or "I couldn't add anyone to this thread.", mention_author=False
    )
    return True


async def handle_member_lookup_request(message: discord.Message) -> bool:
    if message.guild is None or message.mentions:
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
            f"I couldn't find a close match for `{requested_name}` in this server.", mention_author=False
        )
        return True

    member = candidates[0]
    should_add_to_thread = bool(decision["should_add_to_thread"]) and isinstance(
        message.channel, discord.Thread
    )
    PENDING_MEMBER_CONFIRMATIONS[(message.channel.id, message.author.id)] = {
        "candidate_ids": [c.id for c in candidates],
        "current_index": 0,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }

    action = "mention them and add them to this thread" if should_add_to_thread else "mention them"
    await message.reply(
        f"Did you mean {member.mention}? Reply `yes` to {action}, or `no` to cancel.",
        mention_author=False,
    )
    return True


async def handle_solve_request(message: discord.Message) -> bool:
    if not _looks_like_solve_request(message):
        return False

    async with message.channel.typing():
        try:
            reply_text, files = await run_plan_request(message, None)
        except Exception as e:
            traceback.print_exc()
            await message.reply(
                _truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}"),
                mention_author=False,
            )
            return True

    await message.reply(reply_text, files=files, mention_author=False)
    return True


async def handle_workflow_request(message: discord.Message) -> bool:
    text = (message.content or "").strip()
    if not text:
        return False

    direct_request = _detect_artifact_request(text)
    if direct_request is not None:
        action, artifact_type, instruction = direct_request
        try:
            if action == "show":
                reply_text, files = await run_show_artifact_request(message, artifact_type)
            elif action == "undo":
                reply_text, files = await run_undo_request(message, artifact_type)
            elif action == "edit":
                if not instruction:
                    await message.reply(f"Tell me how to revise the {artifact_type}.", mention_author=False)
                    return True
                if artifact_type == "domain":
                    reply_text, files = await run_edit_domain_request(message, instruction)
                elif artifact_type == "problem":
                    reply_text, files = await run_edit_problem_request(message, instruction)
                else:
                    reply_text, files = await run_edit_plan_request(message, instruction)
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
                reply_text, files = await run_plan_request(message, text)
            elif intent == "domain":
                domain_name, domain_text = await run_domain_request(message, text)
                _update_working_artifacts(message, domain=domain_text, domain_name=domain_name)
                reply_text = _format_domain_reply(domain_name, domain_text)
                files = None
            elif intent == "problem":
                problem_name, problem_text = await run_problem_request(message, text)
                _update_working_artifacts(message, problem=problem_text, problem_name=problem_name)
                reply_text = _format_problem_reply(problem_name, problem_text)
                files = None
            elif intent == "validate_plan":
                reply_text = await run_validate_plan_request(message)
                files = None
            elif intent == "validate_domain":
                reply_text = await run_validate_domain_request(message)
                files = None
            elif intent == "validate_task":
                reply_text = await run_validate_task_request(message)
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


async def handle_conversation_message(message: discord.Message) -> None:
    if await handle_member_confirmation_response(message):
        return
    if await handle_thread_add_request(message):
        return
    if await handle_member_lookup_request(message):
        return
    if await handle_solve_request(message):
        return
    if await handle_workflow_request(message):
        return

    text = (message.content or "").strip()
    if not text:
        return

    collab_enabled = is_collab_enabled(str(message.channel.id))
    log_message(
        message.channel.id,
        message.author.id,
        "user",
        _shared_log_content(message) if collab_enabled else text,
        message.guild.id if message.guild else None,
    )

    async with message.channel.typing():
        context = get_recent_context(
            user_id=None if collab_enabled else message.author.id,
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
            shared=collab_enabled,
        )
        selected_model = get_user_model(str(message.author.id)) or MODEL
        try:
            answer = await llm_reply(selected_model, context, user_id=str(message.author.id))
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

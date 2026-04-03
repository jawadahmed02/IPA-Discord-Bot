import ast
import difflib
import json
import re

import discord


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


def _pddl_from_l2p_payload(payload: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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


def _val_output_indicates_valid(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    type_check_failure_patterns = (
        r"type-?checking[^\n]*(?:error|failed|failure|invalid)",
        r"type checking[^\n]*(?:error|failed|failure|invalid)",
    )
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
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in type_check_failure_patterns):
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
        for key in (
            "val.log",
            "pddl_domain.log",
            "pddl_problem.log",
            "pddl_plan.log",
            "stdout",
            "stderr",
        ):
            _add(output.get(key))
        for key, value in output.items():
            if key not in {
                "val.log",
                "pddl_domain.log",
                "pddl_problem.log",
                "pddl_plan.log",
                "stdout",
                "stderr",
            }:
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
                    for subkey in (
                        "val.log",
                        "pddl_domain.log",
                        "pddl_problem.log",
                        "pddl_plan.log",
                        "stdout",
                        "stderr",
                    ):
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

    type_check_failure_patterns = (
        r"type-?checking[^\n]*(?:error|failed|failure|invalid)[^\n]*",
        r"type checking[^\n]*(?:error|failed|failure|invalid)[^\n]*",
    )
    precondition_failure_patterns = (
        r"precondition[^\n]*not satisfied[^\n]*",
        r"precondition[^\n]*is false[^\n]*",
        r"precondition[^\n]*failed[^\n]*",
        r"failed precondition[^\n]*",
    )
    priority_patterns = (
        r"pddl\.[A-Za-z0-9_.]*Error:\s*[^\n]+",
        r"[A-Za-z0-9_.]*Error:\s*[^\n]+",
        r"problem in (?:domain|problem|plan) definition[^\n]*",
        r"unknown type[^\n]*",
        *type_check_failure_patterns,
        r"goal not satisfied[^\n]*",
        r"cannot be applied[^\n]*",
        *precondition_failure_patterns,
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


def _validation_indicates_valid(kind: str, raw: object) -> bool:
    payload = _extract_validation_payload(raw)
    text = _collect_validation_text(payload) or str(raw or "")
    lowered = text.lower()

    type_check_failure_patterns = (
        r"type-?checking[^\n]*(?:error|failed|failure|invalid)",
        r"type checking[^\n]*(?:error|failed|failure|invalid)",
    )
    common_negative_markers = (
        "error:",
        "unknown type",
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
        "cannot be applied",
    )
    plan_negative_patterns = (
        r"precondition[^\n]*not satisfied",
        r"precondition[^\n]*is false",
        r"precondition[^\n]*failed",
        r"failed precondition",
    )

    negative_markers = common_negative_markers + (
        plan_negative_markers if kind == "plan" else ()
    )
    if any(marker in lowered for marker in negative_markers):
        return False
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in type_check_failure_patterns):
        return False
    if kind == "plan" and any(
        re.search(pattern, text, re.IGNORECASE) for pattern in plan_negative_patterns
    ):
        return False

    status = str(payload.get("status", "")).strip().lower()
    if status == "ok":
        return True

    if kind == "plan":
        return _val_output_indicates_valid(text)
    return False


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

    undo_match = re.match(r"^(undo|revert|restore)\s+(the\s+)?(domain|problem)\b", lowered)
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

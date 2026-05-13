"""Tests for IPA_Discbot.bot.parsing — pure functions only, no Discord I/O."""

import pytest

from IPA_Discbot.bot.parsing import (
    _collect_validation_text,
    _detect_artifact_request,
    _extract_validation_payload,
    _normalize_artifact_type,
    _planner_output_indicates_failure,
    _solve_output_has_action_steps,
    _split_discord_message,
    _summarize_validation_failure,
    _truncate_discord_message,
    _val_output_indicates_valid,
    _validation_indicates_valid,
)


# ---------------------------------------------------------------------------
# _truncate_discord_message
# ---------------------------------------------------------------------------

def test_truncate_short_message_unchanged():
    assert _truncate_discord_message("hello") == "hello"


def test_truncate_exact_limit_unchanged():
    msg = "x" * 1900
    assert _truncate_discord_message(msg) == msg


def test_truncate_long_message_adds_ellipsis():
    msg = "x" * 2000
    result = _truncate_discord_message(msg)
    assert len(result) == 1903
    assert result.endswith("...")


# ---------------------------------------------------------------------------
# _split_discord_message
# ---------------------------------------------------------------------------

def test_split_short_message_returns_one_chunk():
    assert _split_discord_message("hello") == ["hello"]


def test_split_long_plain_text():
    line = "a" * 100
    msg = "\n".join([line] * 25)  # 2500 chars
    chunks = _split_discord_message(msg)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1900


def test_split_fenced_code_block_preserves_fence():
    body = "\n".join([f"line {i}: " + "x" * 80 for i in range(30)])
    msg = f"Prefix\n```lisp\n{body}\n```"
    chunks = _split_discord_message(msg)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "```" in chunk


# ---------------------------------------------------------------------------
# _solve_output_has_action_steps
# ---------------------------------------------------------------------------

def test_detects_action_steps():
    text = "; cost = 3\n(move robot room1 room2)\n(pickup robot box1)\n"
    assert _solve_output_has_action_steps(text) is True


def test_no_action_steps():
    assert _solve_output_has_action_steps("Driver aborting: unsolvable") is False
    assert _solve_output_has_action_steps("") is False


# ---------------------------------------------------------------------------
# _planner_output_indicates_failure
# ---------------------------------------------------------------------------

def test_failure_markers_detected():
    assert _planner_output_indicates_failure("Driver aborting after 5s") is True
    assert _planner_output_indicates_failure("Search exit code: 1") is True
    assert _planner_output_indicates_failure("Problem is UNSOLVABLE") is True
    assert _planner_output_indicates_failure("Error: unknown predicate") is True


def test_success_output_not_failure():
    assert _planner_output_indicates_failure("(move robot a b)\n(pickup robot box)") is False


# ---------------------------------------------------------------------------
# _val_output_indicates_valid
# ---------------------------------------------------------------------------

def test_val_positive_markers():
    assert _val_output_indicates_valid("Plan valid\n") is True
    assert _val_output_indicates_valid("Plan executed successfully") is True


def test_val_negative_markers():
    assert _val_output_indicates_valid("Goal not satisfied") is False
    assert _val_output_indicates_valid("precondition failed") is False
    assert _val_output_indicates_valid("Error: unknown type") is False


def test_val_empty_is_invalid():
    assert _val_output_indicates_valid("") is False
    assert _val_output_indicates_valid("   ") is False


# ---------------------------------------------------------------------------
# _normalize_artifact_type
# ---------------------------------------------------------------------------

def test_normalize_valid_types():
    assert _normalize_artifact_type("domain") == "domain"
    assert _normalize_artifact_type("DOMAIN") == "domain"
    assert _normalize_artifact_type("  plan  ") == "plan"
    assert _normalize_artifact_type("problem") == "problem"


def test_normalize_invalid_type_returns_none():
    assert _normalize_artifact_type("action") is None
    assert _normalize_artifact_type("") is None


# ---------------------------------------------------------------------------
# _detect_artifact_request
# ---------------------------------------------------------------------------

def test_detect_show_request():
    result = _detect_artifact_request("show the domain")
    assert result == ("show", "domain", "")

    result = _detect_artifact_request("display plan")
    assert result == ("show", "plan", "")


def test_detect_undo_request():
    result = _detect_artifact_request("undo domain")
    assert result == ("undo", "domain", "")

    result = _detect_artifact_request("revert the problem")
    assert result == ("undo", "problem", "")


def test_detect_edit_request_with_instruction():
    result = _detect_artifact_request("edit domain add a move action")
    assert result is not None
    action, artifact_type, instruction = result
    assert action == "edit"
    assert artifact_type == "domain"
    assert "move" in instruction


def test_detect_no_match_returns_none():
    assert _detect_artifact_request("what is PDDL?") is None
    assert _detect_artifact_request("") is None


# ---------------------------------------------------------------------------
# _extract_validation_payload
# ---------------------------------------------------------------------------

def test_extract_payload_from_dict():
    raw = {"status": "ok", "output": {"stdout": "Plan valid"}}
    payload = _extract_validation_payload(raw)
    assert payload["status"] == "ok"


def test_extract_payload_unwraps_nested_result():
    raw = {"result": {"result": {"status": "ok", "output": {}}}}
    payload = _extract_validation_payload(raw)
    assert payload.get("status") == "ok"


def test_extract_payload_from_string():
    raw = '{"status": "ok"}'
    payload = _extract_validation_payload(raw)
    assert payload.get("status") == "ok"


def test_extract_payload_bad_string_returns_empty():
    payload = _extract_validation_payload("not json at all")
    assert payload == {}


# ---------------------------------------------------------------------------
# _collect_validation_text
# ---------------------------------------------------------------------------

def test_collect_text_from_val_log():
    payload = {"output": {"val.log": "Plan valid\n", "stdout": "done"}}
    text = _collect_validation_text(payload)
    assert "Plan valid" in text
    assert "done" in text


def test_collect_text_deduplicates():
    payload = {"output": {"stdout": "same"}, "stdout": "same"}
    text = _collect_validation_text(payload)
    assert text.count("same") == 1


# ---------------------------------------------------------------------------
# _summarize_validation_failure
# ---------------------------------------------------------------------------

def test_summarize_known_error_pattern():
    details = "problem in domain definition: unknown predicate :foo"
    summary = _summarize_validation_failure(details)
    assert "problem in domain definition" in summary.lower()


def test_summarize_empty_returns_empty():
    assert _summarize_validation_failure("") == ""


# ---------------------------------------------------------------------------
# _validation_indicates_valid
# ---------------------------------------------------------------------------

def test_validation_valid_on_ok_status():
    raw = {"status": "ok"}
    assert _validation_indicates_valid("domain", raw) is True


def test_validation_invalid_on_error_text():
    raw = {"output": {"stdout": "Error: unknown type foo"}}
    assert _validation_indicates_valid("domain", raw) is False


def test_validation_plan_uses_val_markers():
    raw = {"output": {"val.log": "Plan valid\n"}}
    assert _validation_indicates_valid("plan", raw) is True

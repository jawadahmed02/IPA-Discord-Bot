"""Tests for IPA_Discbot.mcp_client.parsing — pure functions only."""

import json

import pytest

from IPA_Discbot.mcp_client.parsing import (
    compact_tool_arguments,
    extract_plan_text,
    extract_val_text,
    format_tool_list,
    parse_solve_response_text,
    require_dict_payload,
)


# ---------------------------------------------------------------------------
# compact_tool_arguments
# ---------------------------------------------------------------------------

def test_compact_removes_none_values():
    args = {"a": 1, "b": None, "c": "hello", "d": None}
    assert compact_tool_arguments(args) == {"a": 1, "c": "hello"}


def test_compact_keeps_falsy_non_none():
    args = {"a": 0, "b": "", "c": False, "d": None}
    result = compact_tool_arguments(args)
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert "d" not in result


# ---------------------------------------------------------------------------
# extract_plan_text
# ---------------------------------------------------------------------------

def test_extract_plan_from_sas_plan():
    payload = {"output": {"sas_plan": "(move robot a b)\n(pickup robot box)"}}
    result = extract_plan_text(payload)
    assert "(move robot a b)" in result


def test_extract_plan_from_nested_result():
    payload = {"result": {"output": {"sas_plan": "(drive truck depot store)"}}}
    result = extract_plan_text(payload)
    assert "(drive truck depot store)" in result


def test_extract_plan_falls_back_to_error():
    payload = {"error": "Planner failed: unsolvable"}
    result = extract_plan_text(payload)
    assert "Planner failed" in result


def test_extract_plan_from_json_string():
    payload_str = json.dumps({"output": {"sas_plan": "(fly plane a b)"}})
    result = extract_plan_text(payload_str)
    assert "(fly plane a b)" in result


def test_extract_plan_plain_text_passthrough():
    result = extract_plan_text("(move a b)\n(drop a)")
    assert "(move a b)" in result


# ---------------------------------------------------------------------------
# extract_val_text
# ---------------------------------------------------------------------------

def test_extract_val_from_val_log():
    payload = {"output": {"val.log": "Plan valid\n"}}
    result = extract_val_text(payload)
    assert "Plan valid" in result


def test_extract_val_from_stdout():
    payload = {"output": {"stdout": "Validation OK"}}
    result = extract_val_text(payload)
    assert "Validation OK" in result


def test_extract_val_empty_string():
    assert extract_val_text("") == ""


def test_extract_val_from_json_string():
    payload_str = json.dumps({"output": {"val.log": "Plan valid\n"}})
    result = extract_val_text(payload_str)
    assert "Plan valid" in result


# ---------------------------------------------------------------------------
# parse_solve_response_text
# ---------------------------------------------------------------------------

def test_parse_solve_extracts_sas_plan():
    text = json.dumps({"output": {"sas_plan": "(move robot a b)"}})
    assert parse_solve_response_text(text) == "(move robot a b)"


def test_parse_solve_plain_text_passthrough():
    text = "(move robot a b)\n(pickup robot box)"
    assert parse_solve_response_text(text) == text


def test_parse_solve_error_field():
    text = json.dumps({"error": "Search failed"})
    assert parse_solve_response_text(text) == "Search failed"


def test_parse_solve_nested_result():
    text = json.dumps({"result": {"output": {"sas_plan": "(fly plane home)"}}})
    assert parse_solve_response_text(text) == "(fly plane home)"


# ---------------------------------------------------------------------------
# require_dict_payload
# ---------------------------------------------------------------------------

def test_require_dict_passes_dict():
    payload = {"domain_pddl": "(define ...)"}
    assert require_dict_payload("update_domain", payload) == payload


def test_require_dict_parses_json_string():
    payload_str = json.dumps({"task_pddl": "(define ...)"})
    result = require_dict_payload("update_task", payload_str)
    assert result["task_pddl"] == "(define ...)"


def test_require_dict_raises_on_non_json_string():
    with pytest.raises(RuntimeError, match="update_domain"):
        require_dict_payload("update_domain", "not json")


def test_require_dict_raises_on_list():
    with pytest.raises(RuntimeError):
        require_dict_payload("update_domain", [1, 2, 3])


# ---------------------------------------------------------------------------
# format_tool_list
# ---------------------------------------------------------------------------

def test_format_tool_list_single():
    assert format_tool_list(("solve",)) == "`solve`"


def test_format_tool_list_two():
    assert format_tool_list(("solve", "validate")) == "`solve`, and `validate`"


def test_format_tool_list_three():
    result = format_tool_list(("a", "b", "c"))
    assert result == "`a`, `b`, and `c`"


def test_format_tool_list_empty():
    assert format_tool_list(()) == "no tools"

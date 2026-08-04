"""Tests for the default search tool scaffold."""

import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from tests.support.command_mocks import (
    captured_args,
    captured_cwd,
    make_executable_available,
    make_executable_missing,
    patch_execution,
)
from tests.support.tool_results import details_of, tool_text
from tile.tools import grep
from tile.tools.grep import GrepDetails
from tile.tools.support import truncation
from tile.types.tools import ToolError


def test_schema_requires_only_pattern() -> None:
    """Require only the search pattern so callers can omit optional controls."""

    assert grep.tool.input_schema["required"] == ["pattern"]


@pytest.mark.parametrize(
    ("kwargs", "expected_argv"),
    [
        pytest.param(
            {
                "pattern": "needle",
                "path": ".",
                "glob": None,
                "ignore_case": False,
                "literal": False,
                "context": 0,
            },
            [
                "--json",
                "--line-number",
                "--color=never",
                "--hidden",
                "--",
                "needle",
                ".",
            ],
            id="defaults",
        ),
        pytest.param(
            {
                "pattern": "needle",
                "path": "src",
                "glob": "**/*.py",
                "ignore_case": True,
                "literal": True,
                "context": 2,
            },
            [
                "--json",
                "--line-number",
                "--color=never",
                "--hidden",
                "--ignore-case",
                "--fixed-strings",
                "--glob",
                "**/*.py",
                "--context",
                "2",
                "--",
                "needle",
                "src",
            ],
            id="all-optional-flags",
        ),
    ],
)
def test_build_args_builds_search_argv(
    kwargs: dict[str, Any],
    expected_argv: list[str],
) -> None:
    """Build argv from default and optional search controls."""

    assert grep._build_args(**kwargs) == expected_argv


@pytest.fixture
def rg_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the rg executable available to fn-level tests."""

    make_executable_available(monkeypatch, "rg", "/usr/bin/rg")


@pytest.fixture
def rg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the rg executable unavailable to fn-level tests."""

    make_executable_missing(monkeypatch)


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch command execution with an async mock for fn-level tests."""

    return patch_execution(monkeypatch, grep)


@pytest.mark.usefixtures("rg_available")
async def test_fn_resolves_search_path_against_supplied_cwd(
    execution: AsyncMock,
    tmp_path: Path,
) -> None:
    """Return compact grep-style text and resolve search roots against the tool cwd."""

    execution.return_value = _event("match", "example.txt", 2, "needle line\n")

    tool_result = await grep.fn(
        grep.GrepInput(pattern="needle", path="src"),
        cwd=tmp_path,
    )

    assert tool_text(tool_result) == "example.txt:2: needle line"
    assert tool_result.details is None
    assert captured_args(execution)[-1] == "src"
    assert captured_cwd(execution) == tmp_path


@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_match_limit_in_details(execution: AsyncMock) -> None:
    """Stop at the match limit, append the notice, and report grep details."""

    execution.return_value = "\n".join(
        [
            _event("match", "one.txt", 1, "needle one\n"),
            _event("match", "two.txt", 2, "needle two\n"),
        ]
    )

    tool_result = await grep.fn(
        grep.GrepInput(pattern="needle", limit=1),
        cwd=Path.cwd(),
    )

    assert tool_text(tool_result) == (
        "one.txt:1: needle one\n\n"
        "[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    )
    details = details_of(tool_result, GrepDetails)
    assert details.match_limit_reached == 1
    assert details.output is not None
    assert details.output.truncated is False
    assert details.lines_truncated is False


@pytest.mark.usefixtures("rg_available")
async def test_fn_clamps_limit_to_one(execution: AsyncMock) -> None:
    """Keep the match limit positive even when callers pass a low limit."""

    execution.return_value = "\n".join(
        [
            _event("match", "one.txt", 1, "needle one\n"),
            _event("match", "two.txt", 2, "needle two\n"),
        ]
    )

    tool_result = await grep.fn(
        grep.GrepInput(pattern="needle", limit=0),
        cwd=Path.cwd(),
    )

    assert tool_text(tool_result) == (
        "one.txt:1: needle one\n\n"
        "[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    )
    assert details_of(tool_result, GrepDetails).match_limit_reached == 1


@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_byte_truncation_in_details(execution: AsyncMock) -> None:
    """Return grep details when formatted output exceeds the byte limit."""

    execution.return_value = "\n".join(
        _event("match", f"{index:03d}.txt", 1, "x" * 196) for index in range(300)
    )

    tool_result = await grep.fn(
        grep.GrepInput(pattern="needle", limit=500),
        cwd=Path.cwd(),
    )

    assert tool_text(tool_result).endswith("\n\n[50.0KB limit reached]")
    details = details_of(tool_result, GrepDetails)
    assert details.match_limit_reached is None
    assert details.lines_truncated is False
    assert details.output is not None
    assert details.output.truncated is True
    assert details.output.truncated_by == "bytes"
    assert details.output.max_bytes == truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.usefixtures("rg_available")
async def test_fn_reports_line_truncation_in_details(execution: AsyncMock) -> None:
    """Shorten long result lines, append the notice, and report grep details."""

    execution.return_value = _event("match", "example.txt", 1, "x" * 501)

    tool_result = await grep.fn(grep.GrepInput(pattern="needle"), cwd=Path.cwd())

    assert tool_text(tool_result) == (
        f"example.txt:1: {'x' * 500}... [truncated]\n\n"
        "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )
    details = details_of(tool_result, GrepDetails)
    assert details.match_limit_reached is None
    assert details.output is not None
    assert details.output.truncated is False
    assert details.lines_truncated is True


@pytest.mark.usefixtures("rg_missing")
async def test_fn_raises_when_command_is_missing() -> None:
    """Raise a clear exception when rg is unavailable."""

    with pytest.raises(ToolError, match="ripgrep"):
        await grep.fn(grep.GrepInput(pattern="needle"), cwd=Path.cwd())


def test_parse_output_ignores_non_search_events() -> None:
    """Ignore events that are not match or context rows."""

    result = grep._parse_output(
        "\n".join(
            [
                json.dumps(
                    {"type": "begin", "data": {"path": {"text": "example.txt"}}}
                ),
                _event("match", "example.txt", 1, "needle\n"),
                json.dumps({"type": "summary", "data": {"elapsed_total": {"secs": 1}}}),
            ]
        ),
        limit=100,
    )

    assert [line.text for line in result.lines] == ["needle"]


def test_build_result_returns_plain_text() -> None:
    """Build grep-style text from raw search output."""

    result = grep._build_result(
        "\n".join(
            [
                _event("context", "example.txt", 1, "before\n"),
                _event("match", "example.txt", 2, "needle line\n"),
                _event("context", "example.txt", 3, "after\n"),
            ]
        ),
        limit=100,
    )

    assert (
        tool_text(result)
        == "example.txt-1- before\nexample.txt:2: needle line\nexample.txt-3- after"
    )


def test_build_result_combines_truncation_notices() -> None:
    """Report match, byte, and line truncation in one notice block."""

    result = grep._build_result(
        "\n".join(
            _event("match", f"{index:03d}.txt", 1, f"{'x' * 501}\n")
            for index in range(120)
        ),
        limit=100,
    )

    assert tool_text(result).endswith(
        "\n\n[100 matches limit reached. Use limit=200 for more, or refine pattern. "
        "50.0KB limit reached. "
        "Some lines truncated to 500 chars. Use read tool to see full lines]"
    )


def _event(
    event_type: Literal["match", "context"],
    path: str,
    line_number: int,
    text: str,
) -> str:
    """Build one JSON event line for parser tests."""

    return json.dumps(
        {
            "type": event_type,
            "data": {
                "path": {"text": path},
                "line_number": line_number,
                "lines": {"text": text},
            },
        }
    )

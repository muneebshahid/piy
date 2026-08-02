"""Tests for the default file path search tool."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import tile.tools.find as find
import tile.tools.support.truncation as truncation
from tile.tools.find import FindDetails
from tile.types.tools import ToolError
from tests.support.command_mocks import (
    captured_args,
    captured_cwd,
    make_executable_available,
    make_executable_missing,
    patch_execution,
)
from tests.support.tool_results import details_of, tool_text


def test_find_schema_requires_only_pattern() -> None:
    """Require only the glob pattern while exposing optional path-search controls."""

    properties = find.tool.input_schema["properties"]

    assert find.tool.name == "find"
    assert find.tool.input_schema["required"] == ["pattern"]
    assert isinstance(properties, dict)
    assert set(properties) == {"pattern", "path", "limit"}


@pytest.fixture
def fd_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fd executable available to fn-level tests."""

    make_executable_available(monkeypatch, "fd", "/usr/bin/fd")


@pytest.fixture
def fd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fd executable unavailable to fn-level tests."""

    make_executable_missing(monkeypatch)


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch command execution with an async mock for fn-level tests."""

    return patch_execution(monkeypatch, find)


@pytest.mark.usefixtures("fd_available")
async def test_fn_uses_default_file_search_flags(
    execution: AsyncMock,
) -> None:
    """Build default fd arguments and return formatted file path results."""

    execution.return_value = "./tile/tools/find.py\n"

    tool_result = await find.fn(find.FindInput(pattern="*.py"), cwd=Path.cwd())
    result = tool_text(tool_result)

    assert result == "tile/tools/find.py"
    assert tool_result.details is None
    execution.assert_awaited_once_with(
        "/usr/bin/fd",
        [
            "--glob",
            "--color=never",
            "--hidden",
            "--no-require-git",
            "--max-results",
            "1001",
            "--",
            "*.py",
            ".",
        ],
        cwd=Path.cwd(),
    )


@pytest.mark.usefixtures("fd_available")
async def test_fn_resolves_search_path_against_supplied_cwd(
    execution: AsyncMock,
    tmp_path: Path,
) -> None:
    """Resolve relative search roots against the supplied tool cwd."""

    execution.return_value = "./tile/tools/find.py\n"

    result = tool_text(
        await find.fn(find.FindInput(pattern="*.py", path="src"), cwd=tmp_path)
    )

    assert result == "tile/tools/find.py"
    assert captured_args(execution)[-1] == "src"
    assert captured_cwd(execution) == tmp_path


@pytest.mark.usefixtures("fd_available")
@pytest.mark.parametrize(
    ("pattern", "effective_pattern"),
    [
        pytest.param("tile/**/*.py", "**/tile/**/*.py", id="path-shaped"),
        pytest.param("/tools/*.py", "**/tools/*.py", id="root-relative"),
        pytest.param("**/tools/*.py", "**/tools/*.py", id="already-prefixed"),
    ],
)
async def test_fn_normalizes_full_path_patterns(
    execution: AsyncMock,
    pattern: str,
    effective_pattern: str,
) -> None:
    """Match path-shaped glob patterns against full candidate paths."""

    execution.return_value = "./tile/tools/find.py\n"

    result = tool_text(
        await find.fn(
            find.FindInput(pattern=pattern, path=".", limit=25),
            cwd=Path.cwd(),
        )
    )

    assert result == "tile/tools/find.py"
    args = captured_args(execution)
    assert "--full-path" in args
    assert args[-3:] == ["--", effective_pattern, "."]


@pytest.mark.usefixtures("fd_available")
async def test_fn_clamps_limit_to_one(execution: AsyncMock) -> None:
    """Keep fd max-results positive even when callers pass a low limit."""

    execution.return_value = "./a.py\n./b.py\n"

    result = tool_text(
        await find.fn(find.FindInput(pattern="*.py", limit=0), cwd=Path.cwd())
    )

    assert (
        result
        == "a.py\n\n[1 results limit reached. Use limit=2 for more, or refine pattern]"
    )
    assert captured_args(execution)[4:6] == ["--max-results", "2"]


@pytest.mark.usefixtures("fd_available")
async def test_fn_returns_no_matches_when_fd_output_is_empty(
    execution: AsyncMock,
) -> None:
    """Return a no-match message when fd emits no paths."""

    execution.return_value = ""

    tool_result = await find.fn(
        find.FindInput(pattern="*.missing"),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result == "No files found matching pattern"
    assert tool_result.details is None


@pytest.mark.usefixtures("fd_missing")
async def test_fn_raises_when_fd_is_missing() -> None:
    """Raise a clear exception when fd is unavailable."""

    with pytest.raises(ToolError, match="fd"):
        await find.fn(find.FindInput(pattern="*.py"), cwd=Path.cwd())


@pytest.mark.usefixtures("fd_available")
async def test_fn_normalizes_paths_and_reports_result_limit(
    execution: AsyncMock,
) -> None:
    """Normalize fd output paths and report when results reach the limit."""

    execution.return_value = (
        ".\\tile\\tools\\find.py\n./tests/test_find_tool.py\n./extra.py\n"
    )

    tool_result = await find.fn(
        find.FindInput(pattern="*.py", limit=2),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result == (
        "tile/tools/find.py\ntests/test_find_tool.py\n\n"
        "[2 results limit reached. Use limit=4 for more, or refine pattern]"
    )
    details = details_of(tool_result, FindDetails)
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.output_lines == 2
    assert details.output.total_lines == 3
    assert details.output.max_lines == 2


@pytest.mark.usefixtures("fd_available")
async def test_fn_reports_byte_limit(execution: AsyncMock) -> None:
    """Append a byte-limit notice when formatted output exceeds 50KB."""

    stdout = "\n".join(f"./{index:03d}-{'x' * 196}.py" for index in range(300))
    execution.return_value = f"{stdout}\n"

    tool_result = await find.fn(
        find.FindInput(pattern="*.py", limit=1000),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)
    notice = "\n\n[50.0KB limit reached]"
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT
    details = details_of(tool_result, FindDetails)
    assert details.output.truncated_by == "bytes"
    assert details.output.output_bytes <= truncation.OUTPUT_BYTE_LIMIT
    assert details.output.total_bytes > truncation.OUTPUT_BYTE_LIMIT


@pytest.mark.usefixtures("fd_available")
async def test_fn_reports_byte_limit_when_byte_boundary_is_first(
    execution: AsyncMock,
) -> None:
    """Report only the byte limit when bytes truncate before result count."""

    stdout = "\n".join(f"./{index:03d}-{'x' * 196}.py" for index in range(300))
    execution.return_value = f"{stdout}\n"

    result = tool_text(
        await find.fn(find.FindInput(pattern="*.py", limit=260), cwd=Path.cwd())
    )

    assert result.endswith("\n\n[50.0KB limit reached]")
    assert "results limit reached" not in result

"""Tests for the default directory listing tool."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.tool_results import details_of, tool_text
from tile.tools import ls
from tile.tools.ls import LsDetails
from tile.tools.support import truncation
from tile.types.tools import ToolError


def test_ls_schema_requires_no_arguments() -> None:
    """Allow callers to omit path and limit."""

    assert ls.tool.input_schema.get("required", []) == []


async def test_ls_returns_all_directory_entries(tmp_path: Path) -> None:
    """Return every file and directory name when the result is under the limit."""

    _create_file(tmp_path / "README.md")
    _create_file(tmp_path / "uv.lock")
    _create_directory(tmp_path / "src")

    tool_result = await ls.fn(
        ls.LsInput(path=str(tmp_path), limit=10),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result.splitlines() == ["README.md", "src/", "uv.lock"]
    assert tool_result.details is None


async def test_ls_resolves_relative_path_against_supplied_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative listing paths against the supplied tool cwd."""

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    _create_file(project / "sample.txt")
    monkeypatch.chdir(other)

    result = tool_text(await ls.fn(ls.LsInput(path=".", limit=10), cwd=project))

    assert result == "sample.txt"


async def test_ls_uses_cwd_when_path_is_omitted(tmp_path: Path) -> None:
    """List the supplied working directory when callers omit path."""

    _create_file(tmp_path / "sample.txt")

    tool_result = await ls.fn(ls.LsInput(limit=10), cwd=tmp_path)
    result = tool_text(tool_result)

    assert result == "sample.txt"
    assert tool_result.details is None


async def test_ls_respects_limit_after_sorting_entries(tmp_path: Path) -> None:
    """Return only the first sorted entries up to the requested limit."""

    _create_file(tmp_path / "b.txt")
    _create_file(tmp_path / "a.txt")
    _create_file(tmp_path / "c.txt")

    tool_result = await ls.fn(
        ls.LsInput(path=str(tmp_path), limit=2),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result.splitlines() == [
        "a.txt",
        "b.txt",
        "",
        "[2 entries limit reached. Use limit=4 for more]",
    ]
    details = details_of(tool_result, LsDetails)
    assert details.output.output_lines == 2
    assert details.output.total_lines == 3
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.keep == "head"
    assert details.output.max_lines == 2


async def test_ls_clamps_limit_to_one(tmp_path: Path) -> None:
    """Keep entry limits positive when callers pass a low limit."""

    _create_file(tmp_path / "b.txt")
    _create_file(tmp_path / "a.txt")
    _create_file(tmp_path / "c.txt")

    result = tool_text(
        await ls.fn(
            ls.LsInput(path=str(tmp_path), limit=0),
            cwd=Path.cwd(),
        )
    )

    assert result.splitlines() == [
        "a.txt",
        "",
        "[1 entries limit reached. Use limit=2 for more]",
    ]


async def test_ls_reports_byte_limit(tmp_path: Path) -> None:
    """Report byte truncation when the listing output exceeds 50KB."""

    _create_long_file_names(tmp_path, count=270)

    tool_result = await ls.fn(
        ls.LsInput(path=str(tmp_path), limit=500),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)
    notice = "\n\n[50.0KB limit reached. Directory has 270 entries]"
    body = result.removesuffix(notice)

    assert result.endswith(notice)
    assert len(body.encode("utf-8")) <= truncation.OUTPUT_BYTE_LIMIT
    details = details_of(tool_result, LsDetails)
    assert details.output.output_lines < details.output.total_lines
    assert details.output.total_lines == 270
    assert details.output.truncated is True
    assert details.output.truncated_by == "bytes"
    assert details.output.max_bytes == truncation.OUTPUT_BYTE_LIMIT
    assert details.output.output_bytes <= truncation.OUTPUT_BYTE_LIMIT
    assert details.output.total_bytes > truncation.OUTPUT_BYTE_LIMIT


async def test_ls_includes_dotfiles_and_dot_directories(tmp_path: Path) -> None:
    """Include hidden files and hidden directories in directory listings."""

    _create_file(tmp_path / ".hidden-file")
    _create_directory(tmp_path / ".hidden-dir")

    result = tool_text(
        await ls.fn(
            ls.LsInput(path=str(tmp_path), limit=10),
            cwd=Path.cwd(),
        )
    )

    assert result.splitlines() == [".hidden-dir/", ".hidden-file"]


async def test_ls_sorts_entries_case_insensitively(tmp_path: Path) -> None:
    """Sort entries alphabetically without separating upper and lower case names."""

    _create_file(tmp_path / "beta.txt")
    _create_file(tmp_path / "Alpha.txt")
    _create_file(tmp_path / "charlie.txt")

    result = tool_text(
        await ls.fn(
            ls.LsInput(path=str(tmp_path), limit=10),
            cwd=Path.cwd(),
        )
    )

    assert result.splitlines() == ["Alpha.txt", "beta.txt", "charlie.txt"]


async def test_ls_reports_empty_directory(tmp_path: Path) -> None:
    """Return an explicit marker for empty directories."""

    tool_result = await ls.fn(
        ls.LsInput(path=str(tmp_path), limit=10),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result == "(empty directory)"
    assert tool_result.details is None


@pytest.mark.parametrize(
    "make_path",
    [
        pytest.param(lambda tmp_path: tmp_path / "missing", id="missing"),
        pytest.param(lambda tmp_path: _regular_file(tmp_path), id="regular-file"),
    ],
)
async def test_ls_raises_for_unlistable_paths(
    tmp_path: Path,
    make_path: Callable[[Path], Path],
) -> None:
    """Raise a tool error when the listing path is missing or not a directory."""

    with pytest.raises(ToolError):
        await ls.fn(
            ls.LsInput(path=str(make_path(tmp_path)), limit=10),
            cwd=Path.cwd(),
        )


def _create_file(path: Path) -> None:
    """Create a small test file."""

    path.write_text("", encoding="utf-8")


def _create_directory(path: Path) -> None:
    """Create a test directory."""

    path.mkdir()


def _create_long_file_names(path: Path, count: int) -> None:
    """Create enough long file names to exceed listing byte limits."""

    for index in range(count):
        _create_file(path / f"{index:03d}-{'x' * 196}.txt")


def _regular_file(tmp_path: Path) -> Path:
    """Create a regular file and return its path."""

    file_path = tmp_path / "file.txt"
    _create_file(file_path)
    return file_path

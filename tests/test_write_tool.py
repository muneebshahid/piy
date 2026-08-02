"""Tests for the default file write tool."""

from pathlib import Path

import pytest

import tile.tools.write as write
from tile.types.tools import ToolError
from tests.support.tool_results import tool_text


def test_write_schema_exposes_write_controls() -> None:
    """Expose only path and content inputs and require both."""

    properties = write.tool.input_schema["properties"]

    assert write.tool.name == "write"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "content"}
    assert write.tool.input_schema["required"] == ["path", "content"]


@pytest.mark.parametrize(
    ("relative_target", "preexisting_content", "content", "expected_bytes"),
    [
        pytest.param("nested/sample.txt", None, "hello", 5, id="nested-fresh"),
        pytest.param("sample.txt", "old", "new", 3, id="overwrite"),
        pytest.param("sample.txt", None, "é", 2, id="multibyte"),
    ],
)
async def test_write_writes_content_and_reports_byte_count(
    tmp_path: Path,
    relative_target: str,
    preexisting_content: str | None,
    content: str,
    expected_bytes: int,
) -> None:
    """Create parents, replace existing content, and report UTF-8 byte counts."""

    file_path = tmp_path / relative_target
    if preexisting_content is not None:
        file_path.write_text(preexisting_content, encoding="utf-8")

    result = tool_text(
        await write.fn(
            write.WriteInput(path=str(file_path), content=content),
            cwd=Path.cwd(),
        )
    )

    assert file_path.read_text(encoding="utf-8") == content
    assert result == f"Successfully wrote {expected_bytes} bytes to {file_path}"


async def test_write_resolves_relative_path_against_supplied_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative write paths against the supplied tool cwd."""

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    monkeypatch.chdir(other)

    result = tool_text(
        await write.fn(
            write.WriteInput(path="relative/sample.txt", content="hello"),
            cwd=project,
        )
    )

    file_path = project / "relative" / "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello"
    assert not (other / "relative" / "sample.txt").exists()
    assert result == f"Successfully wrote 5 bytes to {file_path}"


async def test_write_expands_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand home-directory markers in write paths."""

    monkeypatch.setenv("HOME", str(tmp_path))

    result = tool_text(
        await write.fn(
            write.WriteInput(path="~/sample.txt", content="hello"),
            cwd=Path.cwd(),
        )
    )

    file_path = tmp_path / "sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello"
    assert result == f"Successfully wrote 5 bytes to {file_path}"


async def test_write_raises_when_parent_path_is_file(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark write failures."""

    parent = tmp_path / "parent"
    parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ToolError):
        await write.fn(
            write.WriteInput(path=str(parent / "sample.txt"), content="hello"),
            cwd=Path.cwd(),
        )

"""Tests for the default file read tool."""

import base64
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.tool_results import details_of, tool_text
from tile.tools import read
from tile.tools.read import ReadDetails
from tile.tools.support import truncation
from tile.types.tools import ToolError, ToolImageContent, ToolTextContent


def test_read_schema_exposes_text_read_controls() -> None:
    """Expose the text read inputs, requiring only path with a default limit."""

    properties = read.tool.input_schema["properties"]

    assert read.tool.name == "read"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "offset", "limit"}
    assert properties["limit"]["default"] == truncation.OUTPUT_LINE_LIMIT
    assert read.tool.input_schema["required"] == ["path"]


async def test_read_returns_file_contents(tmp_path: Path) -> None:
    """Return full file contents when no limit is reached."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    tool_result = await read.fn(read.ReadInput(path=str(file_path)), cwd=Path.cwd())
    result = tool_text(tool_result)

    assert result == "one\ntwo\nthree"
    assert tool_result.details is None


async def test_read_resolves_relative_path_against_supplied_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative read paths against the supplied tool cwd."""

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    _write_lines(project / "sample.txt", ["content"])
    monkeypatch.chdir(other)

    result = tool_text(await read.fn(read.ReadInput(path="sample.txt"), cwd=project))

    assert result == "content"


async def test_read_starts_from_one_indexed_offset(tmp_path: Path) -> None:
    """Treat offset as a 1-indexed starting line number."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    result = tool_text(
        await read.fn(read.ReadInput(path=str(file_path), offset=2), cwd=Path.cwd())
    )

    assert result == "two\nthree"


async def test_read_reports_remaining_lines_after_limit(tmp_path: Path) -> None:
    """Report line truncation when a caller limit stops before end of file."""

    file_path = _write_numbered_lines(tmp_path / "sample.txt", count=100)

    tool_result = await read.fn(
        read.ReadInput(path=str(file_path), limit=10),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result == (
        "\n".join(f"line {index}" for index in range(1, 11))
        + "\n\n[Showing lines 1-10 of 100. Use offset=11 to continue.]"
    )
    details = details_of(tool_result, ReadDetails)
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.output_lines == 10
    assert details.output.total_lines == 100
    assert details.output.max_lines == 10


async def test_read_handles_offset_and_limit_together(tmp_path: Path) -> None:
    """Apply offset before applying the caller line limit."""

    file_path = _write_numbered_lines(tmp_path / "sample.txt", count=100)

    tool_result = await read.fn(
        read.ReadInput(path=str(file_path), offset=41, limit=20),
        cwd=Path.cwd(),
    )
    result = tool_text(tool_result)

    assert result == (
        "\n".join(f"line {index}" for index in range(41, 61))
        + "\n\n[Showing lines 41-60 of 100. Use offset=61 to continue.]"
    )
    details = details_of(tool_result, ReadDetails)
    assert details.output.truncated_by == "lines"
    assert details.output.output_lines == 20
    assert details.output.total_lines == 60


async def test_read_clamps_limit_to_one(tmp_path: Path) -> None:
    """Treat a non-positive caller limit as a one-line read."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    result = tool_text(
        await read.fn(read.ReadInput(path=str(file_path), limit=0), cwd=Path.cwd())
    )

    assert result == "one\n\n[Showing lines 1-1 of 3. Use offset=2 to continue.]"


async def test_read_raises_when_offset_is_beyond_file(tmp_path: Path) -> None:
    """Raise a clear error when offset is beyond the file length."""

    file_path = _write_lines(tmp_path / "sample.txt", ["one", "two", "three"])

    with pytest.raises(RuntimeError, match="Offset 100 is beyond end of file"):
        await read.fn(
            read.ReadInput(path=str(file_path), offset=100),
            cwd=Path.cwd(),
        )


async def test_read_reports_byte_truncation(tmp_path: Path) -> None:
    """Append a continuation notice when automatic byte truncation occurs."""

    file_path = _write_lines(tmp_path / "sample.txt", ["x" * 200 for _ in range(500)])

    tool_result = await read.fn(read.ReadInput(path=str(file_path)), cwd=Path.cwd())
    result = tool_text(tool_result)

    assert "50.0KB limit" in result
    assert result.endswith("to continue.]")
    details = details_of(tool_result, ReadDetails)
    assert details.output.truncated_by == "bytes"
    assert details.output.output_bytes <= truncation.OUTPUT_BYTE_LIMIT
    assert details.output.total_bytes > truncation.OUTPUT_BYTE_LIMIT


async def test_read_reports_first_line_exceeds_byte_limit(tmp_path: Path) -> None:
    """Return a bash fallback notice when the first selected line is too large."""

    file_path = _write_lines(tmp_path / "sample.txt", ["x" * (50 * 1024 + 1)])

    tool_result = await read.fn(read.ReadInput(path=str(file_path)), cwd=Path.cwd())
    result = tool_text(tool_result)

    assert result == (
        f"[Line 1 is 50.0KB, exceeds 50.0KB limit. Use bash: "
        f"sed -n '1p' {file_path} | head -c 51200]"
    )
    details = details_of(tool_result, ReadDetails)
    assert details.output.truncated_by == "bytes"
    assert details.output.edge_line_exceeds_limit is True
    assert details.output.output_lines == 0


async def test_read_returns_image_content_for_supported_image(tmp_path: Path) -> None:
    """Return text and base64 image blocks for supported image files."""

    image_bytes = _png_bytes()
    file_path = _write_bytes(tmp_path / "sample.png", image_bytes)

    result = await read.fn(read.ReadInput(path=str(file_path)), cwd=Path.cwd())

    assert len(result.content) == 2
    text_content = result.content[0]
    image_content = result.content[1]
    assert isinstance(text_content, ToolTextContent)
    assert isinstance(image_content, ToolImageContent)
    assert text_content.text == f'<file name="{file_path}">[image/png]</file>'
    assert image_content.mime_type == "image/png"
    assert image_content.data == base64.b64encode(image_bytes).decode("ascii")
    assert result.details is None


async def test_read_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """Raise filesystem errors so the agent can mark tool execution as failed."""

    with pytest.raises(ToolError):
        await read.fn(
            read.ReadInput(path=str(tmp_path / "missing.txt")),
            cwd=Path.cwd(),
        )


async def test_read_expands_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve paths that start with a home-directory marker."""

    monkeypatch.setenv("HOME", str(tmp_path))
    _write_lines(tmp_path / "sample.txt", ["content"])

    result = tool_text(
        await read.fn(read.ReadInput(path="~/sample.txt"), cwd=Path.cwd())
    )

    assert result == "content"


@pytest.mark.parametrize(
    ("stored_name", "spell_request"),
    [
        pytest.param(
            "sample.txt",
            lambda path: f"@{path}",
            id="at-prefix",
        ),
        pytest.param(
            "my file.txt",
            lambda path: str(path).replace(" ", "\u00a0"),
            id="unicode-space",
        ),
        pytest.param(
            "Screenshot 2026-05-28 at 10.30.00\u202fAM.png",
            lambda path: str(path).replace("\u202fAM.", " AM."),
            id="screenshot-ampm-space",
        ),
        pytest.param(
            unicodedata.normalize("NFD", "café.txt"),
            lambda path: str(path.with_name("café.txt")),
            id="nfd-filename",
        ),
        pytest.param(
            "Capture d’ecran.txt",
            lambda path: str(path).replace("’", "'"),
            id="curly-quote",
        ),
        pytest.param(
            unicodedata.normalize("NFD", "Capture d’écran.txt"),
            lambda path: str(path.with_name("Capture d'écran.txt")),
            id="nfd-curly-quote",
        ),
    ],
)
async def test_read_resolves_path_spelling_variants(
    tmp_path: Path,
    stored_name: str,
    spell_request: Callable[[Path], str],
) -> None:
    """Resolve requested spellings that differ from the stored filename."""

    file_path = _write_lines(tmp_path / stored_name, ["content"])

    result = tool_text(
        await read.fn(read.ReadInput(path=spell_request(file_path)), cwd=Path.cwd())
    )

    assert result == "content"


def _write_lines(path: Path, lines: list[str]) -> Path:
    """Write lines to a UTF-8 test file and return its path."""

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_bytes(path: Path, content: bytes) -> Path:
    """Write bytes to a test file and return its path."""

    path.write_bytes(content)
    return path


def _png_bytes() -> bytes:
    """Return enough PNG bytes for MIME sniffing tests."""

    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x00"
        b"IEND"
        b"\xaeB`\x82"
    )


def _write_numbered_lines(path: Path, count: int) -> Path:
    """Write numbered lines to a UTF-8 test file and return its path."""

    return _write_lines(path, [f"line {index}" for index in range(1, count + 1)])

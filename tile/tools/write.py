"""File write tool for the default agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import Field

from tile.tools.support.paths import resolve_to_cwd
from tile.types.tools import ToolDefinition, ToolError, ToolInput, ToolResult


class WriteInput(ToolInput):
    """Model-controlled file write arguments."""

    path: str = Field(description="Path to the file to write, relative or absolute.")
    content: str = Field(description="Content to write to the file.")


async def fn(params: WriteInput, *, cwd: Path) -> ToolResult:
    """Write content to a file."""

    try:
        resolved_path = _resolve_path(params.path, cwd)
        bytes_written = await _execute(resolved_path, params.content)
        return _build_result(bytes_written, resolved_path)
    except (OSError, UnicodeError) as error:
        raise ToolError(str(error)) from error


async def _execute(path: Path, content: str) -> int:
    """Write file content asynchronously."""

    return await asyncio.to_thread(_write_file, path, content)


def _build_result(bytes_written: int, path: Path) -> ToolResult:
    """Build a successful write result."""

    return ToolResult.text(f"Successfully wrote {bytes_written} bytes to {path}")


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a user-provided path for writing."""

    return resolve_to_cwd(path, cwd).resolve(strict=False)


def _write_file(path: Path, content: str) -> int:
    """Create parent directories and write UTF-8 content to a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return len(content.encode("utf-8"))


tool = ToolDefinition(
    name="write",
    description=(
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    ),
    input_model=WriteInput,
    fn=fn,
)

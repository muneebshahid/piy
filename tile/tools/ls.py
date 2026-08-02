"""Directory listing tool for the default agent."""

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field

from tile.tool_truncation import ToolOutputDetails, Truncation
from tile.tools.support.paths import resolve_to_cwd
from tile.tools.support.truncation import (
    OUTPUT_BYTE_LIMIT_LABEL,
    append_notice_block,
    truncate_head,
)
from tile.types.tools import (
    ToolDefinition,
    ToolDetails,
    ToolError,
    ToolInput,
    ToolResult,
)


class LsDetails(ToolDetails):
    """Directory listing metadata for UI and persistence."""

    type: Literal["ls"] = "ls"
    output: ToolOutputDetails


class LsInput(ToolInput):
    """Model-controlled directory listing arguments."""

    path: str = Field(
        default=".",
        description=(
            "The path of the directory to list. Defaults to the current directory."
        ),
    )
    limit: int = Field(
        default=500,
        description="The maximum number of entries to list. Defaults to 500.",
    )


async def fn(params: LsInput, *, cwd: Path) -> ToolResult:
    """List the contents of a directory."""

    try:
        limit = max(1, params.limit)
        resolved_path = _resolve_path(params.path, cwd)
        entries = await _execute(resolved_path)
        return _build_result(entries, limit)
    except OSError as error:
        raise ToolError(str(error)) from error


async def _execute(path: Path) -> list[str]:
    """List directory entries asynchronously."""

    return await asyncio.to_thread(_list_directory_entries, path)


def _build_result(entries: list[str], limit: int) -> ToolResult:
    """Build directory listing results from raw entry names."""

    if not entries:
        return ToolResult.text("(empty directory)")

    truncation = truncate_head("\n".join(entries), max_lines=limit)
    result = truncation.content

    notices: list[str] = []
    if truncation.truncated_by == "lines":
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
    if truncation.truncated_by == "bytes":
        notices.append(
            f"{OUTPUT_BYTE_LIMIT_LABEL} limit reached. "
            f"Directory has {len(entries)} entries"
        )
    result = append_notice_block(result, notices)
    return ToolResult.text(result, details=_build_details(truncation))


def _build_details(truncation: Truncation) -> LsDetails | None:
    """Build ls details when the UI has truncation to render."""

    output_details = ToolOutputDetails.from_truncation(truncation)
    if not output_details.truncated:
        return None
    return LsDetails(output=output_details)


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a directory path against the tool working directory."""

    return resolve_to_cwd(path, cwd).resolve(strict=False)


def _list_directory_entries(path: Path) -> list[str]:
    """Return directory entry names for a path."""

    return sorted(
        (_format_directory_entry(entry) for entry in path.iterdir()),
        key=str.lower,
    )


def _format_directory_entry(entry: Path) -> str:
    """Return a display name with directory entries marked by a slash."""

    if entry.is_dir():
        return f"{entry.name}/"
    return entry.name


tool = ToolDefinition(
    name="ls",
    description="List the contents of a directory.",
    input_model=LsInput,
    fn=fn,
)

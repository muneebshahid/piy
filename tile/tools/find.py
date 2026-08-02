"""File path search tool scaffold for the default agent."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from tile.tool_truncation import ToolOutputDetails, Truncation
from tile.tools.support.executables import execute, require_executable
from tile.tools.support.truncation import (
    OUTPUT_BYTE_LIMIT_LABEL,
    append_notice_block,
    truncate_head,
)
from tile.types.tools import ToolDefinition, ToolDetails, ToolInput, ToolResult


class FindDetails(ToolDetails):
    """File path search metadata for UI and persistence."""

    type: Literal["find"] = "find"
    output: ToolOutputDetails


class FindInput(ToolInput):
    """Model-controlled file path search arguments."""

    pattern: str = Field(
        description=(
            "The glob pattern to match file paths, for example '*.py' or 'src/**/*.py'."
        )
    )
    path: str = Field(
        default=".",
        description="The directory path to search. Defaults to the current directory.",
    )
    limit: int = Field(
        default=1000,
        description="The maximum number of file paths to return. Defaults to 1000.",
    )


async def fn(
    params: FindInput,
    *,
    cwd: Path,
) -> ToolResult:
    """Find file paths matching a glob pattern."""

    limit = max(1, params.limit)
    executable = require_executable("fd", "fd")
    args = _build_args(params.pattern, params.path, limit)
    output = await execute(executable, args, cwd=cwd)
    return _build_result(output, limit)


def _build_args(
    pattern: str,
    path: str,
    limit: int,
) -> list[str]:
    """Build command arguments for a file path search."""

    args = [
        "--glob",
        "--color=never",
        "--hidden",
        "--no-require-git",
        "--max-results",
        str(limit + 1),
    ]

    effective_pattern = _build_effective_pattern(pattern)
    if _matches_full_path(pattern):
        args.append("--full-path")

    args.extend(["--", effective_pattern, path])
    return args


def _build_result(output: str, limit: int) -> ToolResult:
    """Build file path search results from raw fd stdout."""

    paths = [_normalize_path(line) for line in output.splitlines() if line]
    if not paths:
        return ToolResult.text("No files found matching pattern")

    truncation = truncate_head("\n".join(paths), max_lines=limit)
    text = truncation.content

    notices: list[str] = []
    if truncation.truncated_by == "lines":
        notices.append(
            f"{limit} results limit reached. "
            f"Use limit={limit * 2} for more, or refine pattern"
        )
    if truncation.truncated_by == "bytes":
        notices.append(f"{OUTPUT_BYTE_LIMIT_LABEL} limit reached")
    text = append_notice_block(text, notices)
    return ToolResult.text(text, details=_build_details(truncation))


def _build_details(truncation: Truncation) -> FindDetails | None:
    """Build find details when the UI has truncation to render."""

    output_details = ToolOutputDetails.from_truncation(truncation)
    if not output_details.truncated:
        return None
    return FindDetails(output=output_details)


def _build_effective_pattern(pattern: str) -> str:
    """Return the pattern adjusted for fd full-path matching."""

    if not _matches_full_path(pattern) or pattern.startswith("**/"):
        return pattern

    if pattern.startswith("/"):
        return f"**{pattern}"

    return f"**/{pattern}"


def _normalize_path(path: str) -> str:
    """Normalize one fd output path for compact display."""

    normalized = path.rstrip("\r").replace("\\", "/")
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _matches_full_path(pattern: str) -> bool:
    """Return whether a glob pattern should match candidate paths."""

    return "/" in pattern


tool = ToolDefinition(
    name="find",
    description="Search for files by glob pattern.",
    input_model=FindInput,
    fn=fn,
)

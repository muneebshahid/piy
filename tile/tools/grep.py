"""Search tool for the default agent."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from tile.tool_truncation import ToolOutputDetails, Truncation
from tile.tools.support.executables import execute, require_executable
from tile.tools.support.truncation import (
    GREP_LINE_CHARACTER_LIMIT,
    OUTPUT_BYTE_LIMIT_LABEL,
    append_notice_block,
    truncate_head,
    truncate_line,
)
from tile.types.tools import ToolDefinition, ToolDetails, ToolInput, ToolResult


class GrepDetails(ToolDetails):
    """Search metadata for UI and persistence."""

    type: Literal["grep"] = "grep"
    output: ToolOutputDetails
    match_limit_reached: int | None = None
    lines_truncated: bool = False


@dataclass(frozen=True)
class Line:
    """One parsed search output line."""

    kind: Literal["match", "context"]
    path: str
    line_number: int
    text: str


@dataclass(frozen=True)
class Results:
    """Structured search results returned by the search tool."""

    lines: list[Line]
    truncated: bool


class TextValue(BaseModel):
    """Text wrapper used by JSON event fields."""

    text: str


class EventData(BaseModel):
    """Event data fields used by the search tool."""

    path: TextValue
    line_number: int
    lines: TextValue


class SearchEvent(BaseModel):
    """Match or context event shape used by the search tool."""

    type: Literal["match", "context"]
    data: EventData


class GrepInput(ToolInput):
    """Model-controlled file content search arguments."""

    pattern: str = Field(
        description=(
            "The search pattern to find. Treated as a regular expression unless "
            "literal is true."
        )
    )
    path: str = Field(
        default=".",
        description=(
            "The file or directory path to search. Defaults to the current directory."
        ),
    )
    glob: str | None = Field(
        default=None,
        description=(
            "Filter searched files by glob pattern, for example '*.py' or "
            "'**/*_test.py'."
        ),
    )
    ignore_case: bool = Field(
        default=False,
        description="Whether to search case-insensitively. Defaults to false.",
    )
    literal: bool = Field(
        default=False,
        description=(
            "Whether to treat the pattern as a literal string instead of a regular "
            "expression. Defaults to false."
        ),
    )
    context: int = Field(
        default=0,
        description=(
            "The number of lines to include before and after each match. Defaults to 0."
        ),
    )
    limit: int = Field(
        default=100,
        description="The maximum number of matches to return. Defaults to 100.",
    )


async def fn(
    params: GrepInput,
    *,
    cwd: Path,
) -> ToolResult:
    """Search file contents for a pattern."""

    limit = max(1, params.limit)
    executable = require_executable("rg", "ripgrep (rg)")
    args = _build_args(
        params.pattern,
        params.path,
        params.glob,
        ignore_case=params.ignore_case,
        literal=params.literal,
        context=params.context,
    )
    output = await execute(executable, args, allowed_exit_codes=(0, 1), cwd=cwd)
    return _build_result(output, limit)


def _build_result(output: str, limit: int) -> ToolResult:
    """Build search results from raw JSON-lines output."""

    results = _parse_output(output, limit)
    if not results.lines:
        return ToolResult.text("No matches found")

    formatted_lines, line_limit_results = zip(
        *(_format_line(line) for line in results.lines),
        strict=True,
    )

    raw_output = "\n".join(formatted_lines)
    truncation = truncate_head(raw_output, max_lines=len(formatted_lines))
    text = truncation.content
    lines_truncated = any(line_limit_results)

    notices = _build_notices(
        results, limit, truncation, lines_truncated=lines_truncated
    )
    text = append_notice_block(text, notices)

    return ToolResult.text(
        text,
        details=_build_details(
            results, limit, truncation, lines_truncated=lines_truncated
        ),
    )


def _parse_output(output: str, limit: int) -> Results:
    """Parse JSON-lines output into structured search results."""

    lines: list[Line] = []
    match_count = 0
    truncated = False

    for raw_line in output.splitlines():
        parsed_line = _parse_line(raw_line)
        if parsed_line is None:
            continue
        if parsed_line.kind == "match":
            if match_count >= limit:
                truncated = True
                break
            match_count += 1
        lines.append(parsed_line)

    return Results(lines=lines, truncated=truncated)


def _build_notices(
    results: Results,
    limit: int,
    truncation: Truncation,
    *,
    lines_truncated: bool,
) -> list[str]:
    """Build model-visible grep truncation notices."""

    notices: list[str] = []
    if results.truncated:
        notices.append(
            f"{limit} matches limit reached. "
            f"Use limit={limit * 2} for more, or refine pattern"
        )
    if truncation.truncated:
        notices.append(f"{OUTPUT_BYTE_LIMIT_LABEL} limit reached")
    if lines_truncated:
        notices.append(
            f"Some lines truncated to {GREP_LINE_CHARACTER_LIMIT} chars. "
            "Use read tool to see full lines"
        )
    return notices


def _build_details(
    results: Results,
    limit: int,
    truncation: Truncation,
    *,
    lines_truncated: bool,
) -> GrepDetails | None:
    """Build grep details when the UI has a warning to render."""

    output_details = ToolOutputDetails.from_truncation(truncation)
    if not results.truncated and not output_details.truncated and not lines_truncated:
        return None

    return GrepDetails(
        output=output_details,
        match_limit_reached=limit if results.truncated else None,
        lines_truncated=lines_truncated,
    )


def _build_args(
    pattern: str,
    path: str,
    glob: str | None,
    *,
    ignore_case: bool,
    literal: bool,
    context: int,
) -> list[str]:
    """Build command arguments for a search."""

    args = ["--json", "--line-number", "--color=never", "--hidden"]

    if ignore_case:
        args.append("--ignore-case")
    if literal:
        args.append("--fixed-strings")
    if glob:
        args.extend(["--glob", glob])
    if context > 0:
        args.extend(["--context", str(context)])

    args.extend(["--", pattern, path])
    return args


def _parse_line(raw_line: str) -> Line | None:
    """Parse a single JSON event line."""

    try:
        event = SearchEvent.model_validate_json(raw_line)
    except ValidationError:
        return None

    return _build_line(event)


def _build_line(event: SearchEvent) -> Line:
    """Build one search line from event data."""

    return Line(
        kind=event.type,
        path=event.data.path.text,
        line_number=event.data.line_number,
        text=event.data.lines.text.rstrip("\n"),
    )


def _format_line(line: Line) -> tuple[str, bool]:
    """Format one search line using grep's match or context separators."""

    text, was_truncated = truncate_line(line.text)
    if line.kind == "match":
        return f"{line.path}:{line.line_number}: {text}", was_truncated
    return f"{line.path}-{line.line_number}- {text}", was_truncated


tool = ToolDefinition(
    name="grep",
    description="Search file contents for a pattern.",
    input_model=GrepInput,
    fn=fn,
)

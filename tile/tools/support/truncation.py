"""Shared truncation helpers for built-in tool output."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from tile.tool_truncation import Truncation, TruncationKeep, TruncationReason

# Maximum lines to keep before appending a truncation notice.
OUTPUT_LINE_LIMIT: Final[int] = 2000
# Maximum UTF-8 bytes to keep before appending a truncation notice.
OUTPUT_BYTE_LIMIT: Final[int] = 50 * 1024
# Human-readable label for the shared output byte limit.
OUTPUT_BYTE_LIMIT_LABEL: Final[str] = "50.0KB"
# Maximum characters to keep from one grep result text line.
GREP_LINE_CHARACTER_LIMIT: Final[int] = 500


@dataclass(frozen=True)
class TextMeasurements:
    """Line and byte measurements for text being truncated."""

    lines: tuple[str, ...]
    total_lines: int
    total_bytes: int


def append_notice_block(text: str, notices: Sequence[str]) -> str:
    """Append model-visible notices in the standard bracketed block format."""

    if not notices:
        return text
    return f"{text}\n\n[{'. '.join(notices)}]"


def truncate_head(
    text: str,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return leading complete lines constrained by line and byte limits."""

    return truncate_text(text, keep="head", max_lines=max_lines, max_bytes=max_bytes)


def truncate_tail(
    text: str,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return trailing complete lines constrained by line and byte limits."""

    return truncate_text(text, keep="tail", max_lines=max_lines, max_bytes=max_bytes)


def truncate_text(
    text: str,
    *,
    keep: TruncationKeep,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return one edge of text constrained by line and byte limits."""

    measurements = _measure_text(text)
    if _fits_limits(measurements, max_lines, max_bytes):
        return _untruncated_result(text, measurements, keep, max_lines, max_bytes)

    if _edge_line_exceeds_limit(measurements.lines, keep, max_bytes):
        return _edge_line_exceeded_result(measurements, keep, max_lines, max_bytes)

    output_lines, truncated_by = _select_output_lines(
        measurements.lines,
        keep,
        max_lines,
        max_bytes,
    )
    return _truncated_result(
        output_lines,
        truncated_by,
        measurements,
        keep,
        max_lines,
        max_bytes,
    )


def truncate_to_byte_limit(
    text: str,
    byte_limit: int = OUTPUT_BYTE_LIMIT,
) -> tuple[str, bool]:
    """Return text capped to complete lines within the UTF-8 byte limit."""

    if len(text.encode("utf-8")) <= byte_limit:
        return text, False

    output_lines: list[str] = []
    output_bytes = 0
    for line in text.split("\n"):
        separator_bytes = 1 if output_lines else 0
        line_bytes = len(line.encode("utf-8"))
        if output_bytes + separator_bytes + line_bytes > byte_limit:
            break

        output_lines.append(line)
        output_bytes += separator_bytes + line_bytes

    return "\n".join(output_lines), True


def truncate_line(
    line: str,
    character_limit: int = GREP_LINE_CHARACTER_LIMIT,
) -> tuple[str, bool]:
    """Return one line capped to a character limit with a truncation marker."""

    if len(line) <= character_limit:
        return line, False
    return f"{line[:character_limit]}... [truncated]", True


def format_size(byte_count: int) -> str:
    """Format a byte count with compact binary units."""

    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def _measure_text(text: str) -> TextMeasurements:
    """Measure text once for truncation decisions and metadata."""

    lines = tuple(text.split("\n"))
    return TextMeasurements(
        lines=lines,
        total_lines=len(lines),
        total_bytes=len(text.encode("utf-8")),
    )


def _fits_limits(
    measurements: TextMeasurements,
    max_lines: int,
    max_bytes: int,
) -> bool:
    """Return whether text fits within both output limits."""

    return (
        measurements.total_lines <= max_lines and measurements.total_bytes <= max_bytes
    )


def _untruncated_result(
    text: str,
    measurements: TextMeasurements,
    keep: TruncationKeep,
    max_lines: int,
    max_bytes: int,
) -> Truncation:
    """Build truncation metadata for text that already fits."""

    return Truncation(
        content=text,
        truncated=False,
        truncated_by=None,
        keep=keep,
        total_lines=measurements.total_lines,
        total_bytes=measurements.total_bytes,
        output_lines=measurements.total_lines,
        output_bytes=measurements.total_bytes,
        edge_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _edge_line_exceeds_limit(
    lines: Sequence[str],
    keep: TruncationKeep,
    max_bytes: int,
) -> bool:
    """Return whether the retained edge line alone exceeds the byte limit."""

    return len(_edge_line(lines, keep).encode("utf-8")) > max_bytes


def _edge_line(lines: Sequence[str], keep: TruncationKeep) -> str:
    """Return the line at the retained edge."""

    if keep == "head":
        return lines[0]
    return lines[-1]


def _edge_line_exceeded_result(
    measurements: TextMeasurements,
    keep: TruncationKeep,
    max_lines: int,
    max_bytes: int,
) -> Truncation:
    """Build metadata when no complete retained-edge line can fit."""

    return Truncation(
        content="",
        truncated=True,
        truncated_by="bytes",
        keep=keep,
        total_lines=measurements.total_lines,
        total_bytes=measurements.total_bytes,
        output_lines=0,
        output_bytes=0,
        edge_line_exceeds_limit=True,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncated_result(
    output_lines: list[str],
    truncated_by: TruncationReason,
    measurements: TextMeasurements,
    keep: TruncationKeep,
    max_lines: int,
    max_bytes: int,
) -> Truncation:
    """Build metadata for retained output lines."""

    content = "\n".join(output_lines)
    return Truncation(
        content=content,
        truncated=True,
        truncated_by=truncated_by,
        keep=keep,
        total_lines=measurements.total_lines,
        total_bytes=measurements.total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(content.encode("utf-8")),
        edge_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _select_output_lines(
    lines: Sequence[str],
    keep: TruncationKeep,
    max_lines: int,
    max_bytes: int,
) -> tuple[list[str], TruncationReason]:
    """Return selected output lines and the boundary that stopped selection."""

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncationReason = "lines"
    for line in _iter_lines(lines, keep):
        if len(output_lines) >= max_lines:
            break

        separator_bytes = 1 if output_lines else 0
        line_bytes = len(line.encode("utf-8"))
        if output_bytes + separator_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines.append(line)
        output_bytes += separator_bytes + line_bytes

    if keep == "tail":
        output_lines.reverse()
    return output_lines, truncated_by


def _iter_lines(lines: Sequence[str], keep: TruncationKeep) -> Iterator[str]:
    """Iterate lines from the retained edge."""

    if keep == "head":
        return iter(lines)
    return reversed(lines)

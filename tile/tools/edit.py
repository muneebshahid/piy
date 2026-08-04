"""File edit tool for the default agent."""

import asyncio
import difflib
import itertools
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from tile.tools.support.paths import (
    normalize_at_prefix,
    normalize_unicode_spaces,
    resolve_to_cwd,
)
from tile.types.tools import (
    ToolDefinition,
    ToolDetails,
    ToolError,
    ToolInput,
    ToolResult,
)

FUZZY_UNICODE_SPACES = re.compile(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]")
SMART_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201A\u201B]")
SMART_DOUBLE_QUOTES = re.compile(r"[\u201C\u201D\u201E\u201F]")
UNICODE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
BOM = "\ufeff"
type LineEnding = Literal["\n", "\r\n"]


class EditDetails(ToolDetails):
    """File edit metadata for UI and persistence."""

    type: Literal["edit"] = "edit"
    diff: str


class EditReplacement(ToolInput):
    """A single exact text replacement requested by the model."""

    old_text: str = Field(
        description=(
            "Exact text for one targeted replacement. It must be unique in the "
            "original file and must not overlap with any other edits[].old_text in "
            "the same call."
        )
    )
    new_text: str = Field(description="Replacement text for this targeted edit.")


class EditInput(ToolInput):
    """Model-controlled file edit arguments."""

    path: str = Field(description="Path to the file to edit (relative or absolute).")
    edits: list[EditReplacement] = Field(
        min_length=1,
        description=(
            "One or more targeted replacements. Each edit is matched against the "
            "original file, not incrementally. Do not include overlapping or nested "
            "edits. If two changes touch the same block or nearby lines, merge them "
            "into one edit instead."
        ),
    )


async def fn(
    params: EditInput,
    *,
    cwd: Path,
) -> ToolResult:
    """Edit a file with one or more targeted text replacements."""

    try:
        resolved_path = _resolve_path(params.path, cwd)
        result = await _execute(resolved_path, params.edits, params.path)
        return _build_result(result, params.path)
    except (OSError, UnicodeError) as error:
        raise ToolError(str(error)) from error


@dataclass(frozen=True)
class LoadedFile:
    """File content split into mutation metadata and editable text."""

    bom: str
    content: str
    line_ending: LineEnding


@dataclass(frozen=True)
class MatchedSpan:
    """A validated character replacement span in the original content."""

    edit_index: int
    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class EditExecutionResult:
    """Successful edit execution details."""

    path: Path
    replacement_count: int
    diff: str


@dataclass(frozen=True)
class ReplacementApplication:
    """Content before and after applying validated replacements."""

    base_content: str
    new_content: str


class MatchNotFound(ToolError):
    """Raised when a replacement old_text cannot be found."""


async def _execute(
    path: Path,
    replacements: list[EditReplacement],
    display_path: str,
) -> EditExecutionResult:
    """Apply validated replacements to a file asynchronously."""

    return await asyncio.to_thread(_edit_file, path, replacements, display_path)


def _build_result(result: EditExecutionResult, path: str) -> ToolResult:
    """Build a successful edit result."""

    text = f"Successfully replaced {result.replacement_count} block(s) in {path}."
    return ToolResult.text(text, details=EditDetails(diff=result.diff))


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a user path against the tool working directory."""

    normalized_path = normalize_unicode_spaces(normalize_at_prefix(path))
    return resolve_to_cwd(normalized_path, cwd)


def _edit_file(
    path: Path,
    replacements: list[EditReplacement],
    display_path: str,
) -> EditExecutionResult:
    """Read, edit, and write a UTF-8 text file."""

    loaded_file = _load_file(path)
    application = _apply_replacements_with_fallback(
        loaded_file.content,
        replacements,
        display_path,
    )
    _write_loaded_file(path, loaded_file, application.new_content)
    diff = _generate_unified_diff(
        application.base_content,
        application.new_content,
        display_path,
    )
    return EditExecutionResult(
        path=path,
        replacement_count=len(replacements),
        diff=diff,
    )


def _load_file(path: Path) -> LoadedFile:
    """Read a UTF-8 text file and capture BOM and line-ending metadata."""

    raw_content = _read_text(path)
    bom, content = _strip_bom(raw_content)
    return LoadedFile(
        bom=bom,
        content=_normalize_to_lf(content),
        line_ending=_detect_line_ending(content),
    )


def _apply_replacements_with_fallback(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply exact replacements, then retry fuzzy whole-line matching if needed."""

    _validate_non_empty_old_text(replacements, display_path)
    try:
        return _apply_exact_replacements(content, replacements, display_path)
    except MatchNotFound:
        return _apply_fuzzy_replacements(content, replacements, display_path)


def _apply_exact_replacements(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply replacements matched exactly in character space."""

    spans = _match_exact_spans(content, replacements, display_path)
    new_content = _splice_text_spans(content, spans)
    return _build_application(content, new_content, replacements, display_path)


def _apply_fuzzy_replacements(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply replacements matched as fuzzy-normalized whole lines.

    Normalization is used only to locate matching line windows. The windows
    map back to character spans in the original content, so the replacement
    itself is the same character splice the exact path performs.
    """

    spans = _match_fuzzy_spans(content, replacements, display_path)
    new_content = _splice_text_spans(content, spans)
    return _build_application(content, new_content, replacements, display_path)


def _build_application(
    content: str,
    new_content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Reject no-op replacements and pair original with edited content."""

    if content == new_content:
        raise ToolError(_no_change_error(display_path, len(replacements)))
    return ReplacementApplication(base_content=content, new_content=new_content)


def _match_exact_spans(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> list[MatchedSpan]:
    """Find a unique character span for every requested replacement."""

    spans: list[MatchedSpan] = []
    for index, replacement in enumerate(replacements):
        old_text = _normalize_to_lf(replacement.old_text)
        starts = _find_text_starts(content, old_text)
        start = _require_unique_match(starts, display_path, index, len(replacements))
        spans.append(
            MatchedSpan(
                edit_index=index,
                start=start,
                end=start + len(old_text),
                replacement=_normalize_to_lf(replacement.new_text),
            )
        )
    _validate_non_overlapping(spans, display_path)
    return spans


def _match_fuzzy_spans(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> list[MatchedSpan]:
    """Find a unique fuzzy-normalized line window for every requested replacement.

    Lines carry their own trailing newline, so terminators are matched as
    part of each line and a matched window maps directly to its character
    span.
    """

    original_lines = _terminated_lines(content)
    normalized_lines = [_normalize_for_fuzzy_match(line) for line in original_lines]
    line_offsets = _line_start_offsets(original_lines)
    spans: list[MatchedSpan] = []
    for index, replacement in enumerate(replacements):
        window = [
            _normalize_for_fuzzy_match(line)
            for line in _terminated_lines(_normalize_to_lf(replacement.old_text))
        ]
        starts = _find_window_starts(normalized_lines, window)
        start_line = _require_unique_match(
            starts, display_path, index, len(replacements)
        )
        start, end = _window_char_span(
            original_lines,
            line_offsets,
            start_line=start_line,
            last_line=start_line + len(window) - 1,
            old_terminated=window[-1].endswith("\n"),
        )
        spans.append(
            MatchedSpan(
                edit_index=index,
                start=start,
                end=end,
                replacement=_normalize_to_lf(replacement.new_text),
            )
        )
    _validate_non_overlapping(spans, display_path)
    return spans


def _terminated_lines(text: str) -> list[str]:
    """Split text into lines that each keep their own trailing newline."""

    parts = text.split("\n")
    lines = [f"{part}\n" for part in parts[:-1]]
    if parts[-1] != "":
        lines.append(parts[-1])
    return lines


def _line_start_offsets(lines: list[str]) -> list[int]:
    """Return the character offset where each line starts."""

    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    return offsets


def _window_char_span(
    original_lines: list[str],
    line_offsets: list[int],
    *,
    start_line: int,
    last_line: int,
    old_terminated: bool,
) -> tuple[int, int]:
    """Map a matched line window to its character span in the content.

    The span covers the matched lines. When the old text does not claim a
    trailing newline but the matched final line has one, that newline stays
    outside the span.
    """

    start = line_offsets[start_line]
    end = line_offsets[last_line] + len(original_lines[last_line])
    if not old_terminated and original_lines[last_line].endswith("\n"):
        end -= 1
    return start, end


def _find_text_starts(content: str, old_text: str) -> list[int]:
    """Return non-overlapping character offsets where the old text occurs."""

    starts: list[int] = []
    offset = content.find(old_text)
    while offset != -1:
        starts.append(offset)
        offset = content.find(old_text, offset + len(old_text))
    return starts


def _find_window_starts(normalized_lines: list[str], window: list[str]) -> list[int]:
    """Return start indexes where the window matches consecutive lines."""

    return [
        start
        for start in range(len(normalized_lines) - len(window) + 1)
        if _window_matches_at(normalized_lines, window, start)
    ]


def _window_matches_at(
    normalized_lines: list[str],
    window: list[str],
    start: int,
) -> bool:
    """Return whether the window matches the lines starting at one index."""

    *head, last = window
    if normalized_lines[start : start + len(head)] != head:
        return False
    return _last_window_line_matches(normalized_lines[start + len(head)], last)


def _last_window_line_matches(file_line: str, old_line: str) -> bool:
    """Match the window's final line, honoring only a claimed terminator.

    An old line that claims a trailing newline requires one; an old line
    without one also matches a terminated file line, leaving the newline
    outside the replacement span.
    """

    if old_line.endswith("\n"):
        return file_line == old_line
    return file_line == old_line or file_line == f"{old_line}\n"


def _require_unique_match(
    starts: list[int],
    display_path: str,
    edit_index: int,
    total_edits: int,
) -> int:
    """Return the single match start or raise a lookup error."""

    if not starts:
        raise MatchNotFound(_not_found_error(display_path, edit_index, total_edits))
    if len(starts) > 1:
        raise ToolError(
            _duplicate_error(display_path, edit_index, total_edits, len(starts))
        )
    return starts[0]


def _validate_non_overlapping(
    spans: list[MatchedSpan],
    display_path: str,
) -> None:
    """Reject overlapping replacement spans."""

    ordered = sorted(spans, key=lambda span: span.start)
    for previous, current in itertools.pairwise(ordered):
        if previous.end > current.start:
            raise ToolError(
                _overlap_error(previous.edit_index, current.edit_index, display_path)
            )


def _splice_text_spans(content: str, spans: list[MatchedSpan]) -> str:
    """Apply character spans in reverse order so earlier offsets stay stable."""

    result = content
    for span in sorted(spans, key=lambda span: span.start, reverse=True):
        result = result[: span.start] + span.replacement + result[span.end :]
    return result


def _write_loaded_file(path: Path, loaded_file: LoadedFile, content: str) -> None:
    """Write edited content while preserving BOM and line endings."""

    restored_content = _restore_line_endings(content, loaded_file.line_ending)
    _write_text(path, f"{loaded_file.bom}{restored_content}")


def _generate_unified_diff(old_content: str, new_content: str, path: str) -> str:
    """Generate a standard unified diff for edited LF-normalized content."""

    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff_lines)


def _read_text(path: Path) -> str:
    """Read UTF-8 text without newline translation."""

    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text without newline translation."""

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)


def _validate_non_empty_old_text(
    replacements: list[EditReplacement],
    display_path: str,
) -> None:
    """Reject empty old_text values."""

    for index, replacement in enumerate(replacements):
        if replacement.old_text == "":
            raise ToolError(
                _empty_old_text_error(display_path, index, len(replacements))
            )


def _strip_bom(content: str) -> tuple[str, str]:
    """Remove and return a leading UTF-8 BOM marker when present."""

    if content.startswith(BOM):
        return BOM, content[1:]
    return "", content


def _detect_line_ending(content: str) -> LineEnding:
    """Detect whether the first line ending is CRLF or LF."""

    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    if crlf_index < lf_index:
        return "\r\n"
    return "\n"


def _normalize_to_lf(content: str) -> str:
    """Normalize CRLF and CR line endings to LF."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(content: str, line_ending: LineEnding) -> str:
    """Restore LF-normalized content to the original line-ending style."""

    if line_ending == "\r\n":
        return content.replace("\n", "\r\n")
    return content


def _normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fallback matching."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _strip_trailing_line_whitespace(normalized)
    normalized = SMART_SINGLE_QUOTES.sub("'", normalized)
    normalized = SMART_DOUBLE_QUOTES.sub('"', normalized)
    normalized = UNICODE_DASHES.sub("-", normalized)
    return FUZZY_UNICODE_SPACES.sub(" ", normalized)


def _strip_trailing_line_whitespace(text: str) -> str:
    """Strip trailing whitespace from every line."""

    return "\n".join(line.rstrip() for line in text.split("\n"))


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    """Return the not-found edit error message."""

    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match "
            "exactly including all whitespace and newlines. Near matches that "
            "differ in quotes, dashes, or whitespace are only found when the "
            "old text covers whole lines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The old_text must match "
        "exactly including all whitespace and newlines. Near matches that "
        "differ in quotes, dashes, or whitespace are only found when old_text "
        "covers whole lines."
    )


def _duplicate_error(
    path: str,
    edit_index: int,
    total_edits: int,
    occurrences: int,
) -> str:
    """Return the duplicate-match edit error message."""

    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text "
            "must be unique. Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each old_text must be unique. Please provide more context to make it "
        "unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    """Return the empty old_text edit error message."""

    if total_edits == 1:
        return f"old_text must not be empty in {path}."
    return f"edits[{edit_index}].old_text must not be empty in {path}."


def _overlap_error(first_index: int, second_index: int, path: str) -> str:
    """Return the overlapping-edits error message."""

    return (
        f"edits[{first_index}] and edits[{second_index}] overlap in {path}. "
        "Merge them into one edit or target disjoint regions."
    )


def _no_change_error(path: str, total_edits: int) -> str:
    """Return the no-change edit error message."""

    if total_edits == 1:
        return f"No changes made to {path}. The replacement produced identical content."
    return f"No changes made to {path}. The replacements produced identical content."


tool = ToolDefinition(
    name="edit",
    description=(
        "Edit a single file using exact text replacement. Every edits[].old_text "
        "must match a unique, non-overlapping region of the original file. If "
        "two changes affect the same block or nearby lines, merge them into one "
        "edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes."
    ),
    input_model=EditInput,
    fn=fn,
)

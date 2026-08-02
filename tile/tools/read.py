"""Text and image file read tool for the default agent."""

import asyncio
import base64
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from tile.tool_truncation import ToolOutputDetails, Truncation
from tile.tools.support.image_processing import (
    ImageProcessingError,
    ProcessedImage,
    process_image,
)
from tile.tools.support.paths import (
    normalize_at_prefix,
    normalize_unicode_spaces,
    resolve_to_cwd,
)
from tile.tools.support.truncation import (
    OUTPUT_BYTE_LIMIT,
    OUTPUT_BYTE_LIMIT_LABEL,
    OUTPUT_LINE_LIMIT,
    append_notice_block,
    format_size,
    truncate_head,
)
from tile.types.tools import (
    ImageMimeType,
    ToolDefinition,
    ToolDetails,
    ToolError,
    ToolImageContent,
    ToolInput,
    ToolResult,
)

NARROW_NO_BREAK_SPACE = "\u202f"
IMAGE_TYPE_SNIFF_BYTES = 4100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ReadDetails(ToolDetails):
    """File read metadata for UI and persistence."""

    type: Literal["read"] = "read"
    output: ToolOutputDetails


class ReadInput(ToolInput):
    """Model-controlled file read arguments."""

    path: str = Field(description="Path to the file to read, relative or absolute.")
    offset: int | None = Field(
        default=None,
        description="Line number to start reading from, 1-indexed.",
    )
    limit: int = Field(
        default=OUTPUT_LINE_LIMIT,
        description="Maximum number of lines to read.",
    )


@dataclass(frozen=True)
class ReadSelection:
    """Readable file content after offset selection."""

    content: str
    start_line: int
    total_lines: int


async def fn(
    params: ReadInput,
    *,
    cwd: Path,
) -> ToolResult:
    """Read a UTF-8 text file or supported image file."""

    try:
        resolved_path = _resolve_path(params.path, cwd)
        image_mime_type = _detect_supported_image_mime_type(resolved_path)
        if image_mime_type is not None:
            return _read_image(resolved_path, image_mime_type)

        content = await _execute(resolved_path)
        limit = max(1, params.limit)
        return _build_result(content, params.offset, params.path, limit)
    except (OSError, UnicodeError) as error:
        raise ToolError(str(error)) from error


async def _execute(path: Path) -> str:
    """Read a UTF-8 text file from disk."""

    return await asyncio.to_thread(path.read_text, encoding="utf-8")


def _build_result(
    content: str,
    offset: int | None,
    path: str,
    limit: int = OUTPUT_LINE_LIMIT,
) -> ToolResult:
    """Build a text file read result from raw file content."""

    selection = _select_content(content, offset)
    truncation = truncate_head(selection.content, max_lines=limit)
    text = _build_output_text(selection, truncation, path)
    return ToolResult.text(text, details=_build_details(truncation))


def _select_content(content: str, offset: int | None) -> ReadSelection:
    """Select file content from the requested offset through the end."""

    lines = content.split("\n")
    start_index = _start_index(offset)
    if start_index >= len(lines):
        raise ToolError(
            f"Offset {offset} is beyond end of file ({len(lines)} lines total)"
        )

    selected_lines = lines[start_index:]
    return ReadSelection(
        content="\n".join(selected_lines),
        start_line=start_index + 1,
        total_lines=len(lines),
    )


def _build_output_text(
    selection: ReadSelection,
    truncation: Truncation,
    path: str,
) -> str:
    """Build model-visible read text from selected content."""

    if truncation.edge_line_exceeds_limit:
        return _format_first_line_too_large(selection, path)
    if truncation.truncated:
        return _format_truncated_selection(selection, truncation)
    return truncation.content


def _build_details(truncation: Truncation) -> ReadDetails | None:
    """Build read details when the UI has truncation to render."""

    output_details = ToolOutputDetails.from_truncation(truncation)
    if not output_details.truncated:
        return None
    return ReadDetails(output=output_details)


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a path with forgiving user-input path variants."""

    normalized_path = normalize_unicode_spaces(normalize_at_prefix(path))
    resolved = resolve_to_cwd(normalized_path, cwd)
    return _existing_path_variant(resolved)


def _read_image(path: Path, mime_type: ImageMimeType) -> ToolResult:
    """Read an image file and return base64 image content."""

    try:
        processed_image = process_image(path.read_bytes(), mime_type)
    except ImageProcessingError as error:
        return ToolResult.text(f'<file name="{path}">[Image omitted: {error}.]</file>')

    return _format_image_result(path, processed_image)


def _format_image_result(path: Path, image: ProcessedImage) -> ToolResult:
    """Format a processed image as text metadata plus image content."""

    encoded_image = base64.b64encode(image.data).decode("ascii")
    text = _format_image_text(path, image)
    return ToolResult.image(
        text,
        ToolImageContent(data=encoded_image, mime_type=image.mime_type),
    )


def _format_image_text(path: Path, image: ProcessedImage) -> str:
    """Format image metadata for the model-visible text block."""

    content = f"[{image.mime_type}]"
    dimension_note = image.dimension_note()
    if dimension_note is not None:
        content = f"{content}\n{dimension_note}"
    return f'<file name="{path}">{content}</file>'


def _detect_supported_image_mime_type(path: Path) -> ImageMimeType | None:
    """Detect supported image MIME type by sniffing the file header."""

    with path.open("rb") as file:
        header = file.read(IMAGE_TYPE_SNIFF_BYTES)
    return _detect_supported_image_mime_type_from_bytes(header)


def _detect_supported_image_mime_type_from_bytes(
    content: bytes,
) -> ImageMimeType | None:
    """Detect supported image MIME type from leading file bytes."""

    if _is_jpeg(content):
        return "image/jpeg"
    if _is_supported_png(content):
        return "image/png"
    if _is_gif(content):
        return "image/gif"
    if _is_webp(content):
        return "image/webp"
    return None


def _is_jpeg(content: bytes) -> bool:
    """Return whether bytes look like a supported JPEG image."""

    return len(content) >= 4 and content[:3] == b"\xff\xd8\xff" and content[3] != 0xF7


def _is_supported_png(content: bytes) -> bool:
    """Return whether bytes look like a non-animated PNG image."""

    return _is_png(content) and not _is_animated_png(content)


def _is_png(content: bytes) -> bool:
    """Return whether bytes look like a PNG image."""

    return (
        len(content) >= 16
        and content.startswith(PNG_SIGNATURE)
        and content[12:16] == b"IHDR"
    )


def _is_animated_png(content: bytes) -> bool:
    """Return whether PNG bytes contain an animation control chunk before image data."""

    animation_chunk_index = content.find(b"acTL")
    image_data_index = content.find(b"IDAT")
    return animation_chunk_index != -1 and (
        image_data_index == -1 or animation_chunk_index < image_data_index
    )


def _is_gif(content: bytes) -> bool:
    """Return whether bytes look like a GIF image."""

    return content.startswith(b"GIF")


def _is_webp(content: bytes) -> bool:
    """Return whether bytes look like a WEBP image."""

    return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"


def _existing_path_variant(path: Path) -> Path:
    """Return the first existing forgiving path variant."""

    for candidate in _path_variants(path):
        if candidate.exists():
            return candidate
    return path


def _path_variants(path: Path) -> list[Path]:
    """Return path spelling variants to try."""

    macos_screenshot_path = _macos_screenshot_path(path)
    nfd_path = _nfd_path(path)
    curly_quote_path = _curly_quote_path(path)
    nfd_curly_quote_path = _curly_quote_path(nfd_path)
    return [
        path,
        macos_screenshot_path,
        nfd_path,
        curly_quote_path,
        nfd_curly_quote_path,
    ]


def _macos_screenshot_path(path: Path) -> Path:
    """Return a macOS screenshot AM/PM spacing variant."""

    return Path(
        re.sub(
            r" (AM|PM)\.",
            lambda match: f"{NARROW_NO_BREAK_SPACE}{match.group(1)}.",
            str(path),
            flags=re.IGNORECASE,
        )
    )


def _nfd_path(path: Path) -> Path:
    """Return an NFD-normalized path variant."""

    return Path(unicodedata.normalize("NFD", str(path)))


def _curly_quote_path(path: Path) -> Path:
    """Return a path variant using right single quotation marks."""

    return Path(str(path).replace("'", "\u2019"))


def _start_index(offset: int | None) -> int:
    """Convert a 1-indexed offset to a non-negative list index."""

    if offset is None:
        return 0
    return max(0, offset - 1)


def _format_first_line_too_large(selection: ReadSelection, path: str) -> str:
    """Return guidance when the first selected line exceeds the byte limit."""

    first_line = selection.content.split("\n", maxsplit=1)[0]
    first_line_size = format_size(len(first_line.encode("utf-8")))
    return (
        f"[Line {selection.start_line} is {first_line_size}, exceeds "
        f"{OUTPUT_BYTE_LIMIT_LABEL} limit. Use bash: "
        f"sed -n '{selection.start_line}p' {path} | head -c {OUTPUT_BYTE_LIMIT}]"
    )


def _format_truncated_selection(
    selection: ReadSelection, truncation: Truncation
) -> str:
    """Return truncated content with an offset continuation notice."""

    end_line = selection.start_line + truncation.output_lines - 1
    next_offset = end_line + 1
    notice = _truncation_notice(selection, truncation, end_line, next_offset)
    return append_notice_block(truncation.content, [notice])


def _truncation_notice(
    selection: ReadSelection,
    truncation: Truncation,
    end_line: int,
    next_offset: int,
) -> str:
    """Build the continuation notice for automatic truncation."""

    if truncation.truncated_by == "lines":
        return (
            f"Showing lines {selection.start_line}-{end_line} of "
            f"{selection.total_lines}. Use offset={next_offset} to continue."
        )
    return (
        f"Showing lines {selection.start_line}-{end_line} of {selection.total_lines} "
        f"({OUTPUT_BYTE_LIMIT_LABEL} limit). Use offset={next_offset} to continue."
    )


tool = ToolDefinition(
    name="read",
    description=(
        "Read the contents of a UTF-8 text file or supported image file. Text "
        "output is truncated to 2000 lines or 50KB. Use offset and limit for "
        "large text files."
    ),
    input_model=ReadInput,
    fn=fn,
)

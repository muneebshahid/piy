"""Streaming output accumulator for tools that produce unbounded output."""

import codecs
from dataclasses import dataclass, replace

from tile.tool_truncation import Truncation, TruncationReason
from tile.tools.support.truncation import (
    OUTPUT_BYTE_LIMIT,
    OUTPUT_LINE_LIMIT,
    truncate_tail,
)


@dataclass(frozen=True)
class OutputSnapshot:
    """Bounded output content and truncation metadata."""

    content: str
    truncation: Truncation


class OutputAccumulator:
    """Incrementally decode output while keeping a bounded text tail."""

    def __init__(
        self,
        max_lines: int = OUTPUT_LINE_LIMIT,
        max_bytes: int = OUTPUT_BYTE_LIMIT,
    ) -> None:
        self._max_lines = max_lines
        self._max_bytes = max_bytes
        self._max_rolling_bytes = max_bytes * 2
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._tail_buffer = b""
        self._tail_starts_at_line_boundary = True
        self._total_bytes = 0
        self._total_lines = 0
        self._finished = False

    def accumulate(self, chunk: bytes) -> None:
        """Decode and append one raw output chunk."""

        if self._finished:
            raise RuntimeError("Cannot accumulate output after finish")
        self._append_decoded_text(self._decoder.decode(chunk, final=False))

    def finish(self) -> OutputSnapshot:
        """Flush pending decoder state and return the final output snapshot."""

        if not self._finished:
            self._append_decoded_text(self._decoder.decode(b"", final=True))
            self._finished = True
        return self.snapshot()

    def snapshot(self) -> OutputSnapshot:
        """Return the current bounded output snapshot."""

        snapshot_text = self._snapshot_text()
        truncation = truncate_tail(
            snapshot_text,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )
        return OutputSnapshot(
            content=truncation.content,
            truncation=self._with_total_metadata(truncation),
        )

    def _append_decoded_text(self, text: str) -> None:
        """Append decoded text and keep the rolling buffer bounded."""

        if not text:
            return

        text_bytes = text.encode("utf-8")
        self._total_bytes += len(text_bytes)
        if self._total_lines == 0:
            self._total_lines = 1
        self._total_lines += text.count("\n")

        tail_buffer = self._tail_buffer + text_bytes
        start = _trim_start(tail_buffer, self._max_rolling_bytes)
        self._tail_buffer = tail_buffer[start:]
        self._tail_starts_at_line_boundary = _starts_at_line_boundary(
            tail_buffer,
            start,
            previous=self._tail_starts_at_line_boundary,
        )

    def _snapshot_text(self) -> str:
        """Return tail text beginning at a line boundary when possible."""

        tail_text = self._tail_buffer.decode("utf-8", errors="replace")
        if self._tail_starts_at_line_boundary:
            return tail_text

        first_newline = tail_text.find("\n")
        if first_newline == -1:
            return tail_text
        return tail_text[first_newline + 1 :]

    def _with_total_metadata(self, truncation: Truncation) -> Truncation:
        """Overlay global output totals onto a tail truncation result."""

        truncated_by = self._truncated_by(truncation)
        return replace(
            truncation,
            truncated=truncated_by is not None,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_bytes,
        )

    def _truncated_by(self, truncation: Truncation) -> TruncationReason | None:
        """Return the effective truncation boundary for the full stream.

        The supplied truncation only describes the current rolling tail snapshot.
        If that snapshot still exceeds the final output limits, its boundary wins:
        for example, a 100KB retained tail is reduced to 50KB by truncate_tail,
        so this returns "bytes".

        The full stream can also be truncated even when the current snapshot fits.
        For example, the command may emit 3000 lines, but earlier lines are
        dropped by the rolling byte buffer, leaving a 500-line snapshot. In that
        case truncate_tail reports no snapshot truncation, but total_lines still
        exceeds max_lines, so this returns "lines".

        Similarly, a rolling buffer may start in the middle of a long line.
        _snapshot_text removes that partial leading line before truncate_tail
        runs; if the remaining snapshot is under 50KB, the global total_bytes
        check still returns "bytes" so the model sees that earlier output was
        discarded.
        """

        if truncation.truncated_by is not None:
            return truncation.truncated_by
        if self._total_lines > self._max_lines:
            return "lines"
        if self._total_bytes > self._max_bytes:
            return "bytes"
        return None


def _trim_start(content: bytes, max_bytes: int) -> int:
    """Return the byte offset that keeps at most max_bytes from the content tail."""

    if len(content) <= max_bytes:
        return 0
    return _utf8_boundary_at_or_after(content, len(content) - max_bytes)


def _utf8_boundary_at_or_after(content: bytes, start: int) -> int:
    """Return a valid UTF-8 character boundary at or after start."""

    while start < len(content) and (content[start] & 0xC0) == 0x80:
        start += 1
    return start


def _starts_at_line_boundary(
    content: bytes,
    start: int,
    previous: bool,
) -> bool:
    """Return whether a retained suffix begins at a complete line boundary."""

    if start == 0:
        return previous
    return content[start - 1] == 0x0A

"""Tests for streaming tool output accumulation."""

import pytest

from tile.tools.support.output_accumulator import OutputAccumulator


def test_accumulate_decodes_split_utf8_characters() -> None:
    """Decode partial UTF-8 characters across chunk boundaries."""

    output = OutputAccumulator()
    content = "é\nok".encode("utf-8")

    output.accumulate(content[:1])
    output.accumulate(content[1:])
    snapshot = output.finish()

    assert snapshot.content == "é\nok"
    assert snapshot.truncation.truncated is False


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([b"one\ntwo\nthree"], id="single-chunk"),
        pytest.param([b"one\n", b"two\n", b"three"], id="per-line-chunks"),
    ],
)
def test_accumulate_keeps_bounded_tail_with_global_totals(chunks: list[bytes]) -> None:
    """Keep only rolling tail text while preserving full output totals."""

    output = OutputAccumulator(
        max_lines=100,
        max_bytes=6,
    )

    for chunk in chunks:
        output.accumulate(chunk)
    snapshot = output.finish()

    assert snapshot.content == "three"
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "bytes"
    assert snapshot.truncation.total_lines == 3
    assert snapshot.truncation.total_bytes == len(b"".join(chunks))


def test_accumulate_reports_global_line_truncation_when_snapshot_fits() -> None:
    """Report line truncation when rolling trim already dropped earlier lines."""

    max_lines, max_bytes = 2, 10
    output = OutputAccumulator(
        max_lines=max_lines,
        max_bytes=max_bytes,
    )

    output.accumulate(b"aaaaaaaaaaaaaaaaaaaaaaaa\nx\ny")
    snapshot = output.finish()

    assert snapshot.content == "x\ny"
    assert len(snapshot.content.split("\n")) <= max_lines
    assert len(snapshot.content.encode("utf-8")) <= max_bytes
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "lines"
    assert snapshot.truncation.total_lines == 3
    assert snapshot.truncation.output_lines == 2


def test_accumulate_reports_global_byte_truncation_when_snapshot_fits() -> None:
    """Report byte truncation when line-boundary cleanup leaves a small snapshot."""

    max_lines, max_bytes = 100, 10
    output = OutputAccumulator(
        max_lines=max_lines,
        max_bytes=max_bytes,
    )

    output.accumulate(b"aaaaaaaaaaaaaaaaaaaaaaaa\nok")
    snapshot = output.finish()

    assert snapshot.content == "ok"
    assert len(snapshot.content.split("\n")) <= max_lines
    assert len(snapshot.content.encode("utf-8")) <= max_bytes
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "bytes"
    assert snapshot.truncation.total_bytes == 27
    assert snapshot.truncation.output_bytes == 2


def test_accumulate_rejects_chunks_after_finish() -> None:
    """Reject writes after the accumulator has been finalized."""

    output = OutputAccumulator()
    output.finish()

    with pytest.raises(RuntimeError, match="after finish"):
        output.accumulate(b"late")


@pytest.mark.parametrize(
    ("chunk", "expected_total_lines"),
    [
        pytest.param(None, 0, id="no-input"),
        pytest.param(b"x", 1, id="one-byte-no-newline"),
        pytest.param(b"x\n", 2, id="one-byte-with-newline"),
    ],
)
def test_accumulate_total_lines_for_minimal_inputs(
    chunk: bytes | None,
    expected_total_lines: int,
) -> None:
    """Report correct total_lines for empty, single-byte, and newline-terminated input."""

    output = OutputAccumulator()
    if chunk is not None:
        output.accumulate(chunk)
    snapshot = output.finish()

    assert snapshot.truncation.total_lines == expected_total_lines

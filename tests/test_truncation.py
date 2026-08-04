"""Tests for shared tool output truncation helpers."""

from collections.abc import Callable

import pytest

from tile.tool_truncation import Truncation, TruncationReason
from tile.tools.support import truncation


@pytest.mark.parametrize(
    ("notices", "expected"),
    [
        pytest.param([], "content", id="no-notices"),
        pytest.param(
            ["first", "second"], "content\n\n[first. second]", id="joined-notices"
        ),
    ],
)
def test_append_notice_block_formats_notices(notices: list[str], expected: str) -> None:
    """Append notices in the shared bracketed block format, or none at all."""

    assert truncation.append_notice_block("content", notices) == expected


def test_truncate_to_byte_limit_keeps_complete_lines() -> None:
    """Truncate over-limit output at line boundaries instead of mid-line."""

    assert truncation.truncate_to_byte_limit("a.txt\nb.txt", byte_limit=11) == (
        "a.txt\nb.txt",
        False,
    )
    assert truncation.truncate_to_byte_limit("a.txt\nb.txt", byte_limit=10) == (
        "a.txt",
        True,
    )


@pytest.mark.parametrize(
    (
        "truncate",
        "text",
        "max_lines",
        "max_bytes",
        "expected_content",
        "expected_truncated_by",
        "expected_fields",
    ),
    [
        pytest.param(
            truncation.truncate_head,
            "a\nb\nc",
            2,
            100,
            "a\nb",
            "lines",
            {"output_lines": 2, "total_lines": 3},
            id="head-lines",
        ),
        pytest.param(
            truncation.truncate_head,
            "abcd\nefgh",
            100,
            6,
            "abcd",
            "bytes",
            {"output_bytes": 4},
            id="head-bytes",
        ),
        pytest.param(
            truncation.truncate_head,
            "abcdef\nsecond",
            100,
            5,
            "",
            "bytes",
            {"edge_line_exceeds_limit": True, "keep": "head"},
            id="head-first-line-too-long",
        ),
        pytest.param(
            truncation.truncate_head,
            "abcd\nefgh\nijkl",
            2,
            6,
            "abcd",
            "bytes",
            {"output_lines": 1, "total_lines": 3, "total_bytes": 14},
            id="head-both-limits-bytes-bind-first",
        ),
        pytest.param(
            truncation.truncate_tail,
            "a\nb\nc",
            2,
            100,
            "b\nc",
            "lines",
            {"output_lines": 2, "total_lines": 3},
            id="tail-lines",
        ),
        pytest.param(
            truncation.truncate_tail,
            "abcd\nefgh",
            100,
            6,
            "efgh",
            "bytes",
            {"output_bytes": 4},
            id="tail-bytes",
        ),
        pytest.param(
            truncation.truncate_tail,
            "first\nabcdef",
            100,
            5,
            "",
            "bytes",
            {"edge_line_exceeds_limit": True, "keep": "tail"},
            id="tail-last-line-too-long",
        ),
    ],
)
def test_truncate_reports_limit_metadata(
    truncate: Callable[..., Truncation],
    text: str,
    max_lines: int,
    max_bytes: int,
    expected_content: str,
    expected_truncated_by: TruncationReason,
    expected_fields: dict[str, object],
) -> None:
    """Report limit metadata while keeping complete lines at the retained edge."""

    result = truncate(text, max_lines=max_lines, max_bytes=max_bytes)

    assert result.content == expected_content
    assert result.truncated is True
    assert result.truncated_by == expected_truncated_by
    for field, expected in expected_fields.items():
        assert getattr(result, field) == expected

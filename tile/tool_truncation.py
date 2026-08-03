"""Shared data types for tool output helpers and contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel

type TruncationReason = Literal["lines", "bytes"]
type TruncationKeep = Literal["head", "tail"]


@dataclass(frozen=True)
class Truncation:
    """Metadata returned when keeping one edge of tool output."""

    content: str
    truncated: bool
    truncated_by: TruncationReason | None
    keep: TruncationKeep
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    edge_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


class ToolOutputDetails(BaseModel):
    """Structured metadata describing bounded tool output."""

    truncated: bool
    truncated_by: TruncationReason | None
    keep: TruncationKeep
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    edge_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    @classmethod
    def from_truncation(
        cls,
        truncation: Truncation,
    ) -> ToolOutputDetails:
        """Create output details from matching truncation metadata."""

        fields = asdict(truncation)
        del fields["content"]
        return cls(**fields)

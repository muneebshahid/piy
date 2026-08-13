"""Result tool that ends an output-contract run with a declared failure."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tile.result import FAIL_TOOL_NAME
from tile.types.tools import ToolDefinition, ToolDetails, ToolInput, ToolResult


class FailDetails(ToolDetails):
    """Declared failure reason carried on a successful fail execution."""

    type: Literal["fail"] = "fail"
    reason: str


class FailInput(ToolInput):
    """Model-controlled declared failure arguments."""

    reason: str = Field(description="Why the task cannot be completed.")


async def fn(params: FailInput) -> ToolResult:
    """Record the model's reason for not delivering a result."""

    return ToolResult.text(
        "Failure recorded.",
        details=FailDetails(reason=params.reason),
    )


tool = ToolDefinition(
    name=FAIL_TOOL_NAME,
    description=(
        "Report that the task cannot be completed and end the run. "
        "Provide a clear reason naming what is missing or impossible."
    ),
    input_model=FailInput,
    fn=fn,
)

"""Result tool that ends an output-contract run with a validated result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SerializeAsAny

from tile.result import COMPLETE_TOOL_NAME
from tile.types.tools import ToolDefinition, ToolDetails, ToolResult


class CompleteDetails(ToolDetails):
    """Validated run result carried on a successful complete execution."""

    type: Literal["complete"] = "complete"
    value: SerializeAsAny[BaseModel]


def tool(result_type: type[BaseModel]) -> ToolDefinition:
    """Build a complete tool that validates results against one schema."""

    async def fn(params: BaseModel) -> ToolResult:
        """Record the result already validated by the tool executor."""

        return ToolResult.text(
            "Result recorded.",
            details=CompleteDetails(value=params),
        )

    return ToolDefinition(
        name=COMPLETE_TOOL_NAME,
        description=(
            "Report the final result and end the run. The arguments are "
            "validated against the required result schema; validation errors "
            "are returned for correction. Call this exactly once, when the "
            "task is done."
        ),
        input_model=result_type,
        fn=fn,
    )

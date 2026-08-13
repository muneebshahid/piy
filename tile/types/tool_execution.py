"""Runtime contracts for model-requested tool executions."""

from typing import Literal, Self

from pydantic import BaseModel, SerializeAsAny

from tile.types.conversation import ToolResultTurn
from tile.types.tools import ToolDetails, ToolResult, ToolTextContent


class ToolInputIssue(BaseModel):
    """One stable, serializable model-input validation issue."""

    location: tuple[str | int, ...]
    code: str
    message: str


class ToolInputValidationFailure(ToolDetails):
    """Runtime metadata for rejected model-supplied tool arguments."""

    type: Literal["tool_input_validation_failure"] = "tool_input_validation_failure"
    tool_name: str
    issues: list[ToolInputIssue]


class ToolInvocationFailure(ToolDetails):
    """Runtime metadata for an exception raised by a tool implementation."""

    type: Literal["tool_invocation_failure"] = "tool_invocation_failure"
    tool_name: str
    exception_type: str
    message: str


class ToolExecutionOutcome(BaseModel):
    """Full runtime outcome for a tool execution."""

    tool_result_turn: ToolResultTurn
    details: SerializeAsAny[ToolDetails] | None = None

    @classmethod
    def from_result(
        cls,
        *,
        call_id: str,
        tool_name: str,
        result: ToolResult,
    ) -> Self:
        """Build an execution outcome from a raw tool result."""

        return cls(
            tool_result_turn=ToolResultTurn(
                call_id=call_id,
                tool_name=tool_name,
                content=result.content,
            ),
            details=result.details,
        )

    @classmethod
    def from_error(
        cls,
        *,
        call_id: str,
        tool_name: str,
        message: str,
        details: ToolDetails | None = None,
    ) -> Self:
        """Build a model-visible failed execution outcome."""

        return cls(
            tool_result_turn=ToolResultTurn(
                call_id=call_id,
                tool_name=tool_name,
                content=[ToolTextContent(text=message)],
                is_error=True,
            ),
            details=details,
        )

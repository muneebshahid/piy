"""Provider-neutral conversation history models."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field

from tile.types.stream_events import (
    AssistantBlock,
    ProviderSource,
    StopReason,
    StreamDoneEvent,
    StreamErrorEvent,
)
from tile.types.tools import ToolResultContent

type TurnStatus = Literal["completed", "aborted", "error"]


class UserMessage(BaseModel):
    """A user-authored conversation turn."""

    role: Literal["user"] = "user"
    content: str


class AssistantTurn(BaseModel):
    """A finalized assistant turn that can be replayed to a provider."""

    role: Literal["assistant"] = "assistant"
    source: ProviderSource | None = None
    blocks: list[AssistantBlock] = Field(default_factory=list)
    response_id: str | None = None
    stop_reason: StopReason = "stop"
    status: TurnStatus = "completed"
    error_message: str | None = None

    @classmethod
    def from_stream_done(
        cls,
        event: StreamDoneEvent,
    ) -> Self:
        """Create a completed assistant turn from a terminal stream event."""

        return cls(
            source=event.source,
            blocks=[block.model_copy(deep=True) for block in event.blocks],
            response_id=event.response_id,
            stop_reason=event.stop_reason,
            status="completed",
        )

    @classmethod
    def from_stream_error(
        cls,
        event: StreamErrorEvent,
    ) -> Self:
        """Create a failed assistant turn from a terminal stream event."""

        status: TurnStatus = "aborted" if event.stop_reason == "aborted" else "error"
        return cls(
            source=event.source,
            blocks=[block.model_copy(deep=True) for block in event.blocks],
            response_id=event.response_id,
            stop_reason=event.stop_reason,
            status=status,
            error_message=event.error_message,
        )


class ToolResultTurn(BaseModel):
    """A replayable tool result tied to a prior assistant tool call."""

    role: Literal["tool_result"] = "tool_result"
    call_id: str
    tool_name: str
    content: list[ToolResultContent]
    is_error: bool = False


type ConversationItem = Annotated[
    UserMessage | AssistantTurn | ToolResultTurn,
    Field(discriminator="role"),
]

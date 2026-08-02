"""Normalized OpenAI event definitions for the provider pipeline."""

from enum import StrEnum
from typing import Literal, TypedDict

from tile.types.stream_events import StopReason
from tile.types.tools import JsonObject

type Phase = Literal["commentary", "final_answer"]


class NormalizedEventType(StrEnum):
    """Transport-independent event names consumed by the stream assembler."""

    CREATED = "created"
    REASONING_ADDED = "reasoning_added"
    REASONING_DELTA = "reasoning_delta"
    REASONING_DONE = "reasoning_done"
    MESSAGE_ADDED = "message_added"
    MESSAGE_TEXT_DELTA = "message_text_delta"
    MESSAGE_DONE = "message_done"
    TOOL_CALL_ADDED = "tool_call_added"
    TOOL_CALL_ARGUMENTS_DELTA = "tool_call_arguments_delta"
    TOOL_CALL_ARGUMENTS_DONE = "tool_call_arguments_done"
    TOOL_CALL_DONE = "tool_call_done"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


TERMINAL_NORMALIZED_EVENT_TYPES: frozenset[NormalizedEventType] = frozenset(
    {
        NormalizedEventType.COMPLETED,
        NormalizedEventType.INCOMPLETE,
        NormalizedEventType.FAILED,
    }
)


class CreatedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response is created."""

    type: Literal[NormalizedEventType.CREATED]
    response_id: str


class ReasoningAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when a reasoning block starts."""

    type: Literal[NormalizedEventType.REASONING_ADDED]
    item_id: str


class ReasoningDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental reasoning summary text."""

    type: Literal[NormalizedEventType.REASONING_DELTA]
    delta: str


class ReasoningDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when a reasoning block completes."""

    type: Literal[NormalizedEventType.REASONING_DONE]
    item_id: str
    summary_text: str
    reasoning_signature: str | None


class MessageAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when an assistant message block starts."""

    type: Literal[NormalizedEventType.MESSAGE_ADDED]
    item_id: str
    phase: Phase | None


class MessageTextDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental assistant text."""

    type: Literal[NormalizedEventType.MESSAGE_TEXT_DELTA]
    delta: str


class MessageDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when an assistant message block completes."""

    type: Literal[NormalizedEventType.MESSAGE_DONE]
    item_id: str
    text: str
    phase: Phase | None


class ToolCallAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when a tool-call block starts."""

    type: Literal[NormalizedEventType.TOOL_CALL_ADDED]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject


class ToolCallArgumentsDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental tool-call arguments."""

    type: Literal[NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA]
    delta: str


class ToolCallArgumentsDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when full tool-call arguments are available."""

    type: Literal[NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE]
    arguments: JsonObject


class ToolCallDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when a tool-call block completes."""

    type: Literal[NormalizedEventType.TOOL_CALL_DONE]
    provider_item_id: str | None
    call_id: str
    name: str
    arguments: JsonObject


class CompletedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response completes successfully."""

    type: Literal[NormalizedEventType.COMPLETED]
    stop_reason: StopReason


class IncompleteNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response ends incomplete."""

    type: Literal[NormalizedEventType.INCOMPLETE]
    stop_reason: StopReason
    error_message: str | None


class FailedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response fails."""

    type: Literal[NormalizedEventType.FAILED]
    message: str


type NormalizedEvent = (
    CreatedNormalizedEvent
    | ReasoningAddedNormalizedEvent
    | ReasoningDeltaNormalizedEvent
    | ReasoningDoneNormalizedEvent
    | MessageAddedNormalizedEvent
    | MessageTextDeltaNormalizedEvent
    | MessageDoneNormalizedEvent
    | ToolCallAddedNormalizedEvent
    | ToolCallArgumentsDeltaNormalizedEvent
    | ToolCallArgumentsDoneNormalizedEvent
    | ToolCallDoneNormalizedEvent
    | CompletedNormalizedEvent
    | IncompleteNormalizedEvent
    | FailedNormalizedEvent
)

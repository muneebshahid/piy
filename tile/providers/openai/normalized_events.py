"""Normalized OpenAI event definitions for the provider pipeline."""

from enum import StrEnum
from typing import Final, Literal, ReadOnly, TypedDict

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


TERMINAL_NORMALIZED_EVENT_TYPES: Final[frozenset[NormalizedEventType]] = frozenset(
    {
        NormalizedEventType.COMPLETED,
        NormalizedEventType.INCOMPLETE,
        NormalizedEventType.FAILED,
    }
)


class CreatedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response is created."""

    type: ReadOnly[Literal[NormalizedEventType.CREATED]]
    response_id: ReadOnly[str]


class ReasoningAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when a reasoning block starts."""

    type: ReadOnly[Literal[NormalizedEventType.REASONING_ADDED]]
    item_id: ReadOnly[str]


class ReasoningDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental reasoning summary text."""

    type: ReadOnly[Literal[NormalizedEventType.REASONING_DELTA]]
    delta: ReadOnly[str]


class ReasoningDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when a reasoning block completes."""

    type: ReadOnly[Literal[NormalizedEventType.REASONING_DONE]]
    item_id: ReadOnly[str]
    summary_text: ReadOnly[str]
    reasoning_signature: ReadOnly[str | None]


class MessageAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when an assistant message block starts."""

    type: ReadOnly[Literal[NormalizedEventType.MESSAGE_ADDED]]
    item_id: ReadOnly[str]
    phase: ReadOnly[Phase | None]


class MessageTextDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental assistant text."""

    type: ReadOnly[Literal[NormalizedEventType.MESSAGE_TEXT_DELTA]]
    delta: ReadOnly[str]


class MessageDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when an assistant message block completes."""

    type: ReadOnly[Literal[NormalizedEventType.MESSAGE_DONE]]
    item_id: ReadOnly[str]
    text: ReadOnly[str]
    phase: ReadOnly[Phase | None]


class ToolCallAddedNormalizedEvent(TypedDict):
    """Normalized event emitted when a tool-call block starts."""

    type: ReadOnly[Literal[NormalizedEventType.TOOL_CALL_ADDED]]
    provider_item_id: ReadOnly[str | None]
    call_id: ReadOnly[str]
    name: ReadOnly[str]
    arguments: ReadOnly[JsonObject]


class ToolCallArgumentsDeltaNormalizedEvent(TypedDict):
    """Normalized event emitted for incremental tool-call arguments."""

    type: ReadOnly[Literal[NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA]]
    delta: ReadOnly[str]


class ToolCallArgumentsDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when full tool-call arguments are available."""

    type: ReadOnly[Literal[NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE]]
    arguments: ReadOnly[JsonObject]


class ToolCallDoneNormalizedEvent(TypedDict):
    """Normalized event emitted when a tool-call block completes."""

    type: ReadOnly[Literal[NormalizedEventType.TOOL_CALL_DONE]]
    provider_item_id: ReadOnly[str | None]
    call_id: ReadOnly[str]
    name: ReadOnly[str]
    arguments: ReadOnly[JsonObject]


class CompletedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response completes successfully."""

    type: ReadOnly[Literal[NormalizedEventType.COMPLETED]]
    stop_reason: ReadOnly[StopReason]


class IncompleteNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response ends incomplete."""

    type: ReadOnly[Literal[NormalizedEventType.INCOMPLETE]]
    stop_reason: ReadOnly[StopReason]
    error_message: ReadOnly[str | None]


class FailedNormalizedEvent(TypedDict):
    """Normalized event emitted when a provider response fails."""

    type: ReadOnly[Literal[NormalizedEventType.FAILED]]
    message: ReadOnly[str]


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

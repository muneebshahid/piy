"""Normalized OpenAI event factories shared by provider pipeline tests."""

from tile.providers.openai.normalized_events import (
    CompletedNormalizedEvent,
    CreatedNormalizedEvent,
    FailedNormalizedEvent,
    IncompleteNormalizedEvent,
    MessageAddedNormalizedEvent,
    MessageDoneNormalizedEvent,
    MessageTextDeltaNormalizedEvent,
    NormalizedEventType,
    Phase,
    ReasoningAddedNormalizedEvent,
    ReasoningDeltaNormalizedEvent,
    ReasoningDoneNormalizedEvent,
    ToolCallAddedNormalizedEvent,
    ToolCallArgumentsDeltaNormalizedEvent,
    ToolCallArgumentsDoneNormalizedEvent,
    ToolCallDoneNormalizedEvent,
)
from tile.types.stream_events import StopReason
from tile.types.tools import JsonObject


def created_event(response_id: str) -> CreatedNormalizedEvent:
    """Build a created normalized event."""

    return {
        "type": NormalizedEventType.CREATED,
        "response_id": response_id,
    }


def reasoning_added_event(item_id: str) -> ReasoningAddedNormalizedEvent:
    """Build a reasoning-added normalized event."""

    return {
        "type": NormalizedEventType.REASONING_ADDED,
        "item_id": item_id,
    }


def reasoning_delta_event(delta: str) -> ReasoningDeltaNormalizedEvent:
    """Build a reasoning-delta normalized event."""

    return {
        "type": NormalizedEventType.REASONING_DELTA,
        "delta": delta,
    }


def reasoning_done_event(
    item_id: str,
    summary_text: str,
    reasoning_signature: str | None,
) -> ReasoningDoneNormalizedEvent:
    """Build a reasoning-done normalized event."""

    return {
        "type": NormalizedEventType.REASONING_DONE,
        "item_id": item_id,
        "summary_text": summary_text,
        "reasoning_signature": reasoning_signature,
    }


def message_added_event(
    item_id: str,
    phase: Phase | None = None,
) -> MessageAddedNormalizedEvent:
    """Build a message-added normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_ADDED,
        "item_id": item_id,
        "phase": phase,
    }


def message_text_delta_event(
    delta: str,
) -> MessageTextDeltaNormalizedEvent:
    """Build a message text-delta normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
        "delta": delta,
    }


def message_done_event(
    item_id: str,
    text: str,
    phase: Phase | None = None,
) -> MessageDoneNormalizedEvent:
    """Build a message-done normalized event."""

    return {
        "type": NormalizedEventType.MESSAGE_DONE,
        "item_id": item_id,
        "text": text,
        "phase": phase,
    }


def tool_call_added_event(
    provider_item_id: str | None,
    call_id: str,
    name: str,
    arguments: JsonObject,
) -> ToolCallAddedNormalizedEvent:
    """Build a tool-call added normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ADDED,
        "provider_item_id": provider_item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def tool_call_arguments_delta_event(
    delta: str,
) -> ToolCallArgumentsDeltaNormalizedEvent:
    """Build a tool-call arguments delta normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
        "delta": delta,
    }


def tool_call_arguments_done_event(
    arguments: JsonObject,
) -> ToolCallArgumentsDoneNormalizedEvent:
    """Build a tool-call arguments done normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
        "arguments": arguments,
    }


def tool_call_done_event(
    provider_item_id: str | None,
    call_id: str,
    name: str,
    arguments: JsonObject,
) -> ToolCallDoneNormalizedEvent:
    """Build a tool-call done normalized event."""

    return {
        "type": NormalizedEventType.TOOL_CALL_DONE,
        "provider_item_id": provider_item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def completed_event(stop_reason: StopReason) -> CompletedNormalizedEvent:
    """Build a completed normalized event."""

    return {
        "type": NormalizedEventType.COMPLETED,
        "stop_reason": stop_reason,
    }


def incomplete_event(
    stop_reason: StopReason,
    error_message: str | None,
) -> IncompleteNormalizedEvent:
    """Build an incomplete normalized event."""

    return {
        "type": NormalizedEventType.INCOMPLETE,
        "stop_reason": stop_reason,
        "error_message": error_message,
    }


def failed_event(message: str) -> FailedNormalizedEvent:
    """Build a failed normalized event."""

    return {
        "type": NormalizedEventType.FAILED,
        "message": message,
    }

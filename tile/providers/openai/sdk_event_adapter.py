import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import cast

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_message import (
    Content as ResponseMessageContent,
)
from openai.types.responses.response_output_message import (
    ResponseOutputMessage,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)

from tile.providers.openai.normalized_events import (
    CompletedNormalizedEvent,
    CreatedNormalizedEvent,
    FailedNormalizedEvent,
    IncompleteNormalizedEvent,
    MessageAddedNormalizedEvent,
    MessageDoneNormalizedEvent,
    MessageTextDeltaNormalizedEvent,
    NormalizedEvent,
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


async def normalize_sdk_events(
    raw_stream: AsyncGenerator[ResponseStreamEvent],
) -> AsyncGenerator[NormalizedEvent]:
    """Normalize raw SDK events into transport-independent provider events.

    Closing this generator closes ``raw_stream``: closure does not cascade
    through generator chains on its own, so every layer forwards it.
    """

    async with aclosing(raw_stream):
        async for event in raw_stream:
            normalized_event = _normalize_sdk_event(event)
            if normalized_event is not None:
                yield normalized_event


def _normalize_sdk_event(event: ResponseStreamEvent) -> NormalizedEvent | None:
    """Convert one raw SDK event into the shared normalized-event union.

    Reasoning deltas pass through exactly as sent by the provider. Summary
    part boundaries are not surfaced as deltas, so mid-stream text may lack
    the paragraph separators present in the final ``REASONING_DONE`` summary,
    which joins parts with a blank line and is authoritative.
    """

    match event:
        case ResponseCreatedEvent():
            return CreatedNormalizedEvent(
                type=NormalizedEventType.CREATED,
                response_id=event.response.id,
            )
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            return ReasoningAddedNormalizedEvent(
                type=NormalizedEventType.REASONING_ADDED,
                item_id=event.item.id,
            )
        case (
            ResponseReasoningSummaryTextDeltaEvent(delta=delta)
            | ResponseReasoningTextDeltaEvent(delta=delta)
        ):
            return ReasoningDeltaNormalizedEvent(
                type=NormalizedEventType.REASONING_DELTA,
                delta=delta,
            )
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            return ReasoningDoneNormalizedEvent(
                type=NormalizedEventType.REASONING_DONE,
                item_id=event.item.id,
                summary_text=_join_reasoning_summary_text(event.item.summary),
                reasoning_signature=_serialize_reasoning_item(event.item),
            )
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            return MessageAddedNormalizedEvent(
                type=NormalizedEventType.MESSAGE_ADDED,
                item_id=event.item.id,
                phase=_extract_message_phase(event.item),
            )
        case ResponseTextDeltaEvent():
            return MessageTextDeltaNormalizedEvent(
                type=NormalizedEventType.MESSAGE_TEXT_DELTA,
                delta=event.delta,
            )
        case ResponseRefusalDeltaEvent():
            return MessageTextDeltaNormalizedEvent(
                type=NormalizedEventType.MESSAGE_TEXT_DELTA,
                delta=event.delta,
            )
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            return MessageDoneNormalizedEvent(
                type=NormalizedEventType.MESSAGE_DONE,
                item_id=event.item.id,
                text=_join_message_text(event.item.content),
                phase=_extract_message_phase(event.item),
            )
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseFunctionToolCall
        ):
            return ToolCallAddedNormalizedEvent(
                type=NormalizedEventType.TOOL_CALL_ADDED,
                provider_item_id=event.item.id,
                call_id=event.item.call_id,
                name=event.item.name,
                arguments=_parse_tool_call_arguments(event.item.arguments or ""),
            )
        case ResponseFunctionCallArgumentsDeltaEvent():
            return ToolCallArgumentsDeltaNormalizedEvent(
                type=NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
                delta=event.delta,
            )
        case ResponseFunctionCallArgumentsDoneEvent():
            return ToolCallArgumentsDoneNormalizedEvent(
                type=NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
                arguments=_parse_tool_call_arguments(event.arguments),
            )
        case ResponseOutputItemDoneEvent() if isinstance(
            event.item, ResponseFunctionToolCall
        ):
            return ToolCallDoneNormalizedEvent(
                type=NormalizedEventType.TOOL_CALL_DONE,
                provider_item_id=event.item.id,
                call_id=event.item.call_id,
                name=event.item.name,
                arguments=_parse_tool_call_arguments(event.item.arguments or ""),
            )
        case ResponseCompletedEvent():
            return CompletedNormalizedEvent(
                type=NormalizedEventType.COMPLETED,
                stop_reason=_extract_stop_reason(event),
            )
        case ResponseIncompleteEvent():
            return IncompleteNormalizedEvent(
                type=NormalizedEventType.INCOMPLETE,
                stop_reason=_extract_stop_reason(event),
                error_message=_extract_incomplete_error_message(event),
            )
        case ResponseErrorEvent():
            return FailedNormalizedEvent(
                type=NormalizedEventType.FAILED,
                message=_extract_stream_error_message(event),
            )
        case ResponseFailedEvent():
            return FailedNormalizedEvent(
                type=NormalizedEventType.FAILED,
                message=_extract_error_message(event),
            )

    return None


def _parse_tool_call_arguments(arguments: str) -> JsonObject:
    """Parse a JSON arguments string into a dict, returning {} on failure."""

    if not arguments.strip():
        return {}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return cast("JsonObject", parsed)
    return {}


def _extract_error_message(event: ResponseFailedEvent) -> str:
    """Extract the error message from a failed response event."""

    error = getattr(event.response, "error", None)
    if error is None:
        return "OpenAI response failed."

    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message

    return "OpenAI response failed."


def _extract_stream_error_message(event: ResponseErrorEvent) -> str:
    """Extract the error message from a stream error event."""

    return event.message or "OpenAI stream error."


def _extract_incomplete_error_message(event: ResponseIncompleteEvent) -> str:
    """Extract a human-readable error message for an incomplete response."""

    reason = getattr(event.response.incomplete_details, "reason", None)
    if reason == "content_filter":
        return "OpenAI response was truncated by the content filter."
    return "OpenAI response incomplete."


def _extract_stop_reason(
    event: ResponseCompletedEvent | ResponseIncompleteEvent,
) -> StopReason:
    """Derive the stop reason, promoting to tool_use when tool calls are present."""

    base_reason = _extract_base_stop_reason(event)
    if base_reason == "stop" and any(
        isinstance(item, ResponseFunctionToolCall) for item in event.response.output
    ):
        return "tool_use"
    return base_reason


def _extract_base_stop_reason(
    event: ResponseCompletedEvent | ResponseIncompleteEvent,
) -> StopReason:
    """Map the raw response status to a base stop reason."""

    if isinstance(event, ResponseIncompleteEvent):
        reason = getattr(event.response.incomplete_details, "reason", None)
        if reason == "content_filter":
            return "error"
        return "length"
    return "stop"


def _join_reasoning_summary_text(
    summary: Sequence[ResponseReasoningSummary],
) -> str:
    """Join reasoning summary parts into a single paragraph-separated string."""

    return "\n\n".join(item.text for item in summary if item.text)


def _serialize_reasoning_item(item: ResponseReasoningItem) -> str:
    """Serialize a reasoning item to a JSON string for storage."""

    return json.dumps(item.model_dump(mode="json", exclude_none=True))


def _join_message_text(content: Sequence[ResponseMessageContent]) -> str:
    """Concatenate all output-text and refusal parts from a message."""

    parts: list[str] = []
    for item in content:
        if isinstance(item, ResponseOutputText):
            parts.append(item.text)
        elif isinstance(item, ResponseOutputRefusal):
            parts.append(item.refusal)
    return "".join(parts)


def _extract_message_phase(
    item: ResponseOutputMessage,
) -> Phase | None:
    """Extract the structured output phase from a message item if present."""

    phase = getattr(item, "phase", None)
    if phase in {"commentary", "final_answer"}:
        return phase
    return None

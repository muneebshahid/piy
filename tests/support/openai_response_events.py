"""Raw OpenAI response event factories for provider-boundary tests."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from unittest.mock import AsyncMock

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from tests.support.async_streams import async_stream
from tile.types.tools import JsonObject


@dataclass
class FakeResponsesEndpoint:
    """Fake OpenAI responses endpoint used by provider integration tests."""

    create: AsyncMock


@dataclass
class FakeOpenAIClient:
    """Fake OpenAI client exposing the responses endpoint contract."""

    responses: FakeResponsesEndpoint


def build_fake_openai_client(
    events: Sequence[ResponseStreamEvent],
) -> FakeOpenAIClient:
    """Build a fake client whose responses stream yields raw events."""

    return FakeOpenAIClient(
        responses=FakeResponsesEndpoint(
            create=AsyncMock(return_value=FakeRawResponseStream(events))
        )
    )


class FakeRawResponseStream:
    """Fake SDK response stream exposing the ``AsyncStream`` contract.

    Tracks closure so provider tests can assert the transport is released.
    """

    def __init__(self, events: Sequence[ResponseStreamEvent]) -> None:
        """Wrap static raw events behind the SDK stream surface."""

        self._events = async_stream(events)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        """Iterate the wrapped raw events."""

        return self._events

    async def close(self) -> None:
        """Record closure and release the wrapped event source."""

        self.closed = True
        await self._events.aclose()


def raw_response_stream(
    events: Sequence[ResponseStreamEvent],
) -> AsyncGenerator[ResponseStreamEvent]:
    """Yield raw OpenAI response events as the provider's wrapper does."""

    return async_stream(events)


def response_created_event(
    sequence_number: int,
    response_id: str,
) -> ResponseCreatedEvent:
    """Build a raw response-created event."""

    return ResponseCreatedEvent.model_validate(
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "in_progress"),
        }
    )


def response_completed_event(
    sequence_number: int,
    response_id: str,
    *,
    output: Sequence[JsonObject] | None = None,
) -> ResponseCompletedEvent:
    """Build a raw response-completed event."""

    return ResponseCompletedEvent.model_validate(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": _response_payload(response_id, "completed", output=output),
        }
    )


def response_failed_event(
    sequence_number: int,
    response_id: str,
    message: str,
) -> ResponseFailedEvent:
    """Build a raw response-failed event."""

    return ResponseFailedEvent.model_validate(
        {
            "type": "response.failed",
            "sequence_number": sequence_number,
            "response": _response_payload(
                response_id,
                "failed",
                error={"code": "server_error", "message": message},
            ),
        }
    )


def response_error_event(
    sequence_number: int,
    message: str,
) -> ResponseErrorEvent:
    """Build a raw transport error event."""

    return ResponseErrorEvent.model_validate(
        {
            "type": "error",
            "sequence_number": sequence_number,
            "code": "server_error",
            "message": message,
            "param": None,
        }
    )


def response_incomplete_event(
    sequence_number: int,
    response_id: str,
    reason: str,
    *,
    output: Sequence[JsonObject] | None = None,
) -> ResponseIncompleteEvent:
    """Build a raw response-incomplete event."""

    return ResponseIncompleteEvent.model_validate(
        {
            "type": "response.incomplete",
            "sequence_number": sequence_number,
            "response": _response_payload(
                response_id,
                "incomplete",
                output=output,
                incomplete_reason=reason,
            ),
        }
    )


def reasoning_added_event(
    sequence_number: int,
    item_id: str,
    *,
    output_index: int = 0,
) -> ResponseOutputItemAddedEvent:
    """Build a raw reasoning item added event."""

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "summary": [],
                "status": "in_progress",
            },
        }
    )


def reasoning_summary_part_added_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryPartAddedEvent:
    """Build a raw reasoning-summary part added event."""

    return ResponseReasoningSummaryPartAddedEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "part": {"type": "summary_text", "text": ""},
            "summary_index": summary_index,
        }
    )


def reasoning_summary_delta_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    delta: str,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryTextDeltaEvent:
    """Build a raw reasoning-summary text delta event."""

    return ResponseReasoningSummaryTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "delta": delta,
        }
    )


def reasoning_text_delta_event(
    sequence_number: int,
    item_id: str,
    content_index: int,
    delta: str,
    *,
    output_index: int = 0,
) -> ResponseReasoningTextDeltaEvent:
    """Build a raw reasoning-text delta event."""

    return ResponseReasoningTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        }
    )


def reasoning_summary_part_done_event(
    sequence_number: int,
    item_id: str,
    summary_index: int,
    text: str,
    *,
    output_index: int = 0,
) -> ResponseReasoningSummaryPartDoneEvent:
    """Build a raw reasoning-summary part done event."""

    return ResponseReasoningSummaryPartDoneEvent.model_validate(
        {
            "type": "response.reasoning_summary_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "part": {"type": "summary_text", "text": text},
            "summary_index": summary_index,
        }
    )


def reasoning_done_event(
    sequence_number: int,
    item_id: str,
    summary_texts: Sequence[str],
    *,
    output_index: int = 0,
) -> ResponseOutputItemDoneEvent:
    """Build a raw reasoning item done event."""

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": text} for text in summary_texts
                ],
                "status": "completed",
            },
        }
    )


def message_added_event(
    sequence_number: int,
    item_id: str,
    *,
    output_index: int = 1,
    phase: str | None = None,
) -> ResponseOutputItemAddedEvent:
    """Build a raw assistant message item added event."""

    item: JsonObject = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    if phase is not None:
        item["phase"] = phase

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": item,
        }
    )


def content_part_added_event(
    sequence_number: int,
    item_id: str,
    content_kind: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseContentPartAddedEvent:
    """Build a raw assistant content part added event."""

    return ResponseContentPartAddedEvent.model_validate(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": _content_part(content_kind),
        }
    )


def text_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseTextDeltaEvent:
    """Build a raw assistant output text delta event."""

    return ResponseTextDeltaEvent.model_validate(
        {
            "type": "response.output_text.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
            "logprobs": [],
        }
    )


def refusal_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
    content_index: int = 0,
) -> ResponseRefusalDeltaEvent:
    """Build a raw assistant refusal delta event."""

    return ResponseRefusalDeltaEvent.model_validate(
        {
            "type": "response.refusal.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
        }
    )


def message_done_event(
    sequence_number: int,
    item_id: str,
    content: Sequence[JsonObject],
    *,
    output_index: int = 1,
    phase: str | None = None,
) -> ResponseOutputItemDoneEvent:
    """Build a raw assistant message item done event."""

    item: JsonObject = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": list(content),
    }
    if phase is not None:
        item["phase"] = phase

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": item,
        }
    )


def function_tool_call_added_event(
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    *,
    arguments: str = "",
    output_index: int = 1,
) -> ResponseOutputItemAddedEvent:
    """Build a raw function-tool-call item added event."""

    return ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            },
        }
    )


def function_tool_call_arguments_delta_event(
    sequence_number: int,
    item_id: str,
    delta: str,
    *,
    output_index: int = 1,
) -> ResponseFunctionCallArgumentsDeltaEvent:
    """Build a raw function-tool-call arguments delta event."""

    return ResponseFunctionCallArgumentsDeltaEvent.model_validate(
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta,
        }
    )


def function_tool_call_arguments_done_event(
    sequence_number: int,
    item_id: str,
    arguments: str,
    *,
    name: str = "get_weather",
    output_index: int = 1,
) -> ResponseFunctionCallArgumentsDoneEvent:
    """Build a raw function-tool-call arguments done event."""

    return ResponseFunctionCallArgumentsDoneEvent.model_validate(
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item_id": item_id,
            "name": name,
            "arguments": arguments,
        }
    )


def function_tool_call_done_event(
    sequence_number: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    *,
    output_index: int = 1,
) -> ResponseOutputItemDoneEvent:
    """Build a raw function-tool-call item done event."""

    return ResponseOutputItemDoneEvent.model_validate(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            },
        }
    )


def _response_payload(
    response_id: str,
    status: str,
    *,
    output: Sequence[JsonObject] | None = None,
    error: dict[str, str] | None = None,
    incomplete_reason: str | None = None,
) -> JsonObject:
    """Build a minimal OpenAI response payload for event model validation."""

    return {
        "id": response_id,
        "created_at": 0.0,
        "error": error,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason is not None else None
        ),
        "model": "gpt-5.4",
        "object": "response",
        "output": list(output or []),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }


def _content_part(content_kind: str) -> JsonObject:
    """Build a raw assistant content part payload."""

    if content_kind == "output_text":
        return {"type": "output_text", "text": "", "annotations": []}
    if content_kind == "refusal":
        return {"type": "refusal", "refusal": ""}
    return {"type": content_kind, "text": "internal"}

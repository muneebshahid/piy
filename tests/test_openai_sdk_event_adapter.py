"""Tests for raw OpenAI SDK event normalization."""

import json
from dataclasses import dataclass

import pytest
from openai.types.responses import ResponseStreamEvent

from tile.providers.openai.normalized_events import NormalizedEvent, NormalizedEventType
from tile.providers.openai.sdk_event_adapter import normalize_sdk_events
from tests.support import normalized_events as normalized
from tests.support.openai_response_events import (
    content_part_added_event,
    function_tool_call_added_event,
    function_tool_call_arguments_delta_event,
    function_tool_call_arguments_done_event,
    function_tool_call_done_event,
    message_added_event,
    message_done_event,
    raw_response_stream,
    reasoning_added_event,
    reasoning_done_event,
    reasoning_summary_delta_event,
    reasoning_summary_part_added_event,
    reasoning_summary_part_done_event,
    reasoning_text_delta_event,
    refusal_delta_event,
    response_completed_event,
    response_created_event,
    response_error_event,
    response_failed_event,
    response_incomplete_event,
    text_delta_event,
)


@dataclass(frozen=True)
class NormalizationCase:
    """Defines one raw SDK event and its expected normalized provider event."""

    name: str
    raw_event: ResponseStreamEvent
    expected_event: NormalizedEvent


def _build_normalization_cases() -> list[NormalizationCase]:
    """Returns the full raw-event normalization matrix for the adapter."""

    return [
        *_build_lifecycle_cases(),
        *_build_reasoning_cases(),
        *_build_message_cases(),
        *_build_tool_call_cases(),
        *_build_failure_cases(),
    ]


def _build_lifecycle_cases() -> list[NormalizationCase]:
    """Builds normalization cases for stream lifecycle events."""

    return [
        *_build_response_start_cases(),
        *_build_response_completion_cases(),
        *_build_response_incomplete_cases(),
    ]


def _build_response_start_cases() -> list[NormalizationCase]:
    """Builds normalization cases for response creation events."""

    return [
        NormalizationCase(
            name="response.created",
            raw_event=response_created_event(
                sequence_number=1, response_id="resp_created"
            ),
            expected_event=normalized.created_event("resp_created"),
        ),
    ]


def _build_response_completion_cases() -> list[NormalizationCase]:
    """Builds normalization cases for completed response events."""

    return [
        NormalizationCase(
            name="response.completed.stop",
            raw_event=response_completed_event(
                sequence_number=2,
                response_id="resp_completed",
            ),
            expected_event=normalized.completed_event("stop"),
        ),
        NormalizationCase(
            name="response.completed.tool_use",
            raw_event=response_completed_event(
                sequence_number=3,
                response_id="resp_tool_use",
                output=[
                    {
                        "id": "fc_123",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city":"Berlin"}',
                    }
                ],
            ),
            expected_event=normalized.completed_event("tool_use"),
        ),
    ]


def _build_response_incomplete_cases() -> list[NormalizationCase]:
    """Builds normalization cases for incomplete response events."""

    return [
        NormalizationCase(
            name="response.incomplete.length",
            raw_event=response_incomplete_event(
                sequence_number=4,
                response_id="resp_incomplete",
                reason="max_output_tokens",
            ),
            expected_event=normalized.incomplete_event(
                "length", "OpenAI response incomplete."
            ),
        ),
        NormalizationCase(
            name="response.incomplete.content_filter",
            raw_event=response_incomplete_event(
                sequence_number=5,
                response_id="resp_filtered",
                reason="content_filter",
            ),
            expected_event=normalized.incomplete_event(
                "error", "OpenAI response was truncated by the content filter."
            ),
        ),
    ]


def _build_reasoning_cases() -> list[NormalizationCase]:
    """Builds normalization cases for reasoning-related events."""

    return [
        *_build_reasoning_start_cases(),
        *_build_reasoning_delta_cases(),
        *_build_reasoning_done_cases(),
    ]


def _build_reasoning_start_cases() -> list[NormalizationCase]:
    """Builds normalization cases for reasoning start events."""

    return [
        NormalizationCase(
            name="response.output_item.added.reasoning",
            raw_event=reasoning_added_event(
                sequence_number=6,
                item_id="rs_added",
            ),
            expected_event=normalized.reasoning_added_event("rs_added"),
        ),
    ]


def _build_reasoning_delta_cases() -> list[NormalizationCase]:
    """Builds normalization cases for reasoning delta events."""

    return [
        NormalizationCase(
            name="response.reasoning_summary_text.delta",
            raw_event=reasoning_summary_delta_event(
                sequence_number=7,
                item_id="rs_delta",
                summary_index=0,
                delta="Thinking...",
            ),
            expected_event=normalized.reasoning_delta_event("Thinking..."),
        ),
        NormalizationCase(
            name="response.reasoning_text.delta",
            raw_event=reasoning_text_delta_event(
                sequence_number=8,
                item_id="rs_delta",
                content_index=0,
                delta="Thinking with text...",
            ),
            expected_event=normalized.reasoning_delta_event("Thinking with text..."),
        ),
    ]


def _build_reasoning_done_cases() -> list[NormalizationCase]:
    """Builds normalization cases for completed reasoning items."""

    return [
        NormalizationCase(
            name="response.output_item.done.reasoning",
            raw_event=reasoning_done_event(
                sequence_number=9,
                item_id="rs_done",
                summary_texts=["step one", "step two"],
            ),
            expected_event=normalized.reasoning_done_event(
                item_id="rs_done",
                summary_text="step one\n\nstep two",
                reasoning_signature=json.dumps(
                    {
                        "id": "rs_done",
                        "summary": [
                            {"text": "step one", "type": "summary_text"},
                            {"text": "step two", "type": "summary_text"},
                        ],
                        "type": "reasoning",
                        "status": "completed",
                    }
                ),
            ),
        ),
    ]


def _build_message_cases() -> list[NormalizationCase]:
    """Builds normalization cases for assistant message events."""

    return [
        *_build_message_start_cases(),
        *_build_message_delta_cases(),
        *_build_message_done_cases(),
    ]


def _build_message_start_cases() -> list[NormalizationCase]:
    """Builds normalization cases for assistant message start events."""

    return [
        NormalizationCase(
            name="response.output_item.added.message",
            raw_event=message_added_event(
                sequence_number=10,
                item_id="msg_added",
                output_index=0,
                phase="commentary",
            ),
            expected_event=normalized.message_added_event(
                "msg_added", phase="commentary"
            ),
        ),
    ]


def _build_message_delta_cases() -> list[NormalizationCase]:
    """Builds normalization cases for assistant message delta events."""

    return [
        NormalizationCase(
            name="response.output_text.delta",
            raw_event=text_delta_event(
                sequence_number=14,
                item_id="msg_text_delta",
                delta="Hello",
                output_index=0,
            ),
            expected_event=normalized.message_text_delta_event("Hello"),
        ),
        NormalizationCase(
            name="response.refusal.delta",
            raw_event=refusal_delta_event(
                sequence_number=15,
                item_id="msg_refusal_delta",
                delta="No",
                output_index=0,
            ),
            expected_event=normalized.message_text_delta_event("No"),
        ),
    ]


def _build_message_done_cases() -> list[NormalizationCase]:
    """Builds normalization cases for completed assistant message events."""

    return [
        NormalizationCase(
            name="response.output_item.done.message",
            raw_event=message_done_event(
                sequence_number=16,
                item_id="msg_done",
                output_index=0,
                phase="final_answer",
                content=[
                    {"type": "output_text", "text": "Hello", "annotations": []},
                    {"type": "refusal", "refusal": " there"},
                ],
            ),
            expected_event=normalized.message_done_event(
                "msg_done", "Hello there", phase="final_answer"
            ),
        ),
    ]


def _build_tool_call_cases() -> list[NormalizationCase]:
    """Builds normalization cases for tool-call related events."""

    return [
        *_build_tool_call_start_cases(),
        *_build_tool_call_argument_cases(),
        *_build_tool_call_done_cases(),
    ]


def _build_tool_call_start_cases() -> list[NormalizationCase]:
    """Builds normalization cases for started tool-call events."""

    return [
        NormalizationCase(
            name="response.output_item.added.function_call",
            raw_event=function_tool_call_added_event(
                sequence_number=17,
                item_id="fc_added",
                call_id="call_added",
                name="get_weather",
                arguments='{"city":"Berlin"}',
                output_index=0,
            ),
            expected_event=normalized.tool_call_added_event(
                provider_item_id="fc_added",
                call_id="call_added",
                name="get_weather",
                arguments={"city": "Berlin"},
            ),
        ),
        NormalizationCase(
            name="response.output_item.added.function_call.blank_arguments",
            raw_event=function_tool_call_added_event(
                sequence_number=18,
                item_id="fc_added_blank",
                call_id="call_added_blank",
                name="get_weather",
                arguments="",
                output_index=0,
            ),
            expected_event=normalized.tool_call_added_event(
                provider_item_id="fc_added_blank",
                call_id="call_added_blank",
                name="get_weather",
                arguments={},
            ),
        ),
    ]


def _build_tool_call_argument_cases() -> list[NormalizationCase]:
    """Builds normalization cases for streamed tool-call arguments."""

    return [
        NormalizationCase(
            name="response.function_call_arguments.delta",
            raw_event=function_tool_call_arguments_delta_event(
                sequence_number=19,
                item_id="fc_delta",
                delta='{"city"',
                output_index=0,
            ),
            expected_event=normalized.tool_call_arguments_delta_event('{"city"'),
        ),
        NormalizationCase(
            name="response.function_call_arguments.done",
            raw_event=function_tool_call_arguments_done_event(
                sequence_number=20,
                item_id="fc_args_done",
                name="get_weather",
                arguments='{"city":"Berlin"}',
                output_index=0,
            ),
            expected_event=normalized.tool_call_arguments_done_event(
                {"city": "Berlin"}
            ),
        ),
        NormalizationCase(
            name="response.function_call_arguments.done.malformed_arguments",
            raw_event=function_tool_call_arguments_done_event(
                sequence_number=21,
                item_id="fc_args_done_malformed",
                name="get_weather",
                arguments='{"city"',
                output_index=0,
            ),
            expected_event=normalized.tool_call_arguments_done_event({}),
        ),
    ]


def _build_tool_call_done_cases() -> list[NormalizationCase]:
    """Builds normalization cases for completed tool-call events."""

    return [
        NormalizationCase(
            name="response.output_item.done.function_call",
            raw_event=function_tool_call_done_event(
                sequence_number=22,
                item_id="fc_done",
                call_id="call_done",
                name="get_weather",
                arguments='{"city":"Berlin"}',
                output_index=0,
            ),
            expected_event=normalized.tool_call_done_event(
                provider_item_id="fc_done",
                call_id="call_done",
                name="get_weather",
                arguments={"city": "Berlin"},
            ),
        ),
        NormalizationCase(
            name="response.output_item.done.function_call.non_object_arguments",
            raw_event=function_tool_call_done_event(
                sequence_number=23,
                item_id="fc_done_non_object",
                call_id="call_done_non_object",
                name="get_weather",
                arguments='["Berlin"]',
                output_index=0,
            ),
            expected_event=normalized.tool_call_done_event(
                provider_item_id="fc_done_non_object",
                call_id="call_done_non_object",
                name="get_weather",
                arguments={},
            ),
        ),
    ]


def _build_failure_cases() -> list[NormalizationCase]:
    """Builds normalization cases for stream failure events."""

    return [
        NormalizationCase(
            name="error",
            raw_event=response_error_event(
                sequence_number=24,
                message="Socket closed",
            ),
            expected_event=normalized.failed_event("Socket closed"),
        ),
        NormalizationCase(
            name="response.failed",
            raw_event=response_failed_event(
                sequence_number=25,
                response_id="resp_failed",
                message="Model overloaded",
            ),
            expected_event=normalized.failed_event("Model overloaded"),
        ),
    ]


@pytest.mark.parametrize(
    "case",
    _build_normalization_cases(),
    ids=lambda case: case.name,
)
async def test_normalize_sdk_events_maps_each_supported_raw_event(
    case: NormalizationCase,
) -> None:
    """Normalizes every supported raw SDK event into the expected provider event."""

    events = [
        event
        async for event in normalize_sdk_events(raw_response_stream([case.raw_event]))
    ]

    assert events == [case.expected_event]


@pytest.mark.parametrize(
    "ignored_event",
    [
        reasoning_summary_part_added_event(1, "rs_ignored", summary_index=0),
        reasoning_summary_part_done_event(2, "rs_ignored", 0, "step"),
        content_part_added_event(3, "msg_1", "output_text", output_index=0),
        content_part_added_event(4, "msg_2", "refusal", output_index=0),
        content_part_added_event(5, "msg_3", "reasoning_text", output_index=0),
    ],
    ids=[
        "summary_part_added",
        "summary_part_done",
        "content_part_output_text",
        "content_part_refusal",
        "content_part_unsupported",
    ],
)
async def test_normalize_sdk_events_ignores_unmapped_raw_events(
    ignored_event: ResponseStreamEvent,
) -> None:
    """Drop raw SDK events that have no normalized mapping."""

    events = [
        event
        async for event in normalize_sdk_events(raw_response_stream([ignored_event]))
    ]

    assert events == []


async def test_normalize_sdk_events_passes_summary_parts_through_without_separators() -> (
    None
):
    """Forward reasoning deltas verbatim and drop summary part boundaries.

    Part boundaries are not synthesized into the delta stream; the
    REASONING_DONE summary joins parts with a blank line and is the
    authoritative text.
    """

    raw_events = [
        reasoning_summary_delta_event(1, "rs_2", 0, "Part one"),
        reasoning_summary_part_done_event(2, "rs_2", 0, "Part one"),
        reasoning_summary_delta_event(3, "rs_2", 1, "Part two"),
        reasoning_summary_part_done_event(4, "rs_2", 1, "Part two"),
        reasoning_done_event(5, "rs_2", ["Part one", "Part two"]),
    ]

    events = [
        event async for event in normalize_sdk_events(raw_response_stream(raw_events))
    ]

    assert events[:2] == [
        {"type": NormalizedEventType.REASONING_DELTA, "delta": "Part one"},
        {"type": NormalizedEventType.REASONING_DELTA, "delta": "Part two"},
    ]
    done_event = events[2]
    assert done_event["type"] == NormalizedEventType.REASONING_DONE
    assert done_event["summary_text"] == "Part one\n\nPart two"  # type: ignore[typeddict-item]
    assert len(events) == 3

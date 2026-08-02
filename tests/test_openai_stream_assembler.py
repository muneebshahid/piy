"""Tests for assembling normalized provider events into stream events.

These tests document the middle of the streaming lifecycle. OpenAI transport
adapters produce normalized events such as ``CREATED``, ``MESSAGE_TEXT_DELTA``,
and ``COMPLETED``. The stream assembler consumes those events, privately
accumulates assistant blocks, and emits provider stream events such as
``text_start``, ``text_delta``, ``text_end``, and ``stream_done``.
"""

from collections.abc import Sequence

import pytest

from tile.providers.openai.normalized_events import NormalizedEvent
from tile.providers.openai.stream_assembler import assemble_stream
from tile.types.stream_events import (
    ProviderSource,
    ProviderStreamEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from tests.support.async_streams import async_stream
from tests.support.normalized_events import (
    completed_event,
    created_event,
    failed_event,
    incomplete_event,
    message_added_event,
    message_done_event,
    message_text_delta_event,
    reasoning_added_event,
    reasoning_delta_event,
    reasoning_done_event,
    tool_call_added_event,
    tool_call_arguments_delta_event,
    tool_call_arguments_done_event,
    tool_call_done_event,
)
from tests.support.stream_assertions import (
    expect_metadata_string as _expect_metadata_string,
    expect_reasoning_block as _expect_reasoning_block,
    expect_stream_event as _expect_event_type,
    expect_text_block as _expect_text_block,
    expect_tool_call_block as _expect_tool_call_block,
)


async def test_assemble_stream_accumulates_reasoning_and_text_blocks() -> None:
    """Accumulates reasoning and text blocks onto the terminal stream event."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(_reasoning_text_events()),
            source=_source(),
        )
    ]

    _assert_reasoning_text_event_sequence(events)
    _assert_reasoning_text_stream_content(events)
    _assert_reasoning_text_terminal_blocks(events)


async def test_assemble_stream_preserves_reasoning_deltas_when_done_summary_is_empty() -> (
    None
):
    """Preserves accumulated reasoning deltas when the done event has no summary."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(
                [
                    created_event("resp_reasoning_empty_done"),
                    reasoning_added_event("rs_123"),
                    reasoning_delta_event("Draft summary"),
                    reasoning_done_event(
                        item_id="rs_123",
                        summary_text="",
                        reasoning_signature='{"id":"rs_123"}',
                    ),
                    completed_event("stop"),
                ]
            ),
            source=_source(),
        )
    ]

    reasoning_end = _expect_event_type(events[3], ReasoningEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)
    reasoning_block = _expect_reasoning_block(reasoning_end.block)
    done_reasoning_block = _expect_reasoning_block(done.blocks[0])

    assert [event.type for event in events] == [
        "stream_start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_end",
        "stream_done",
    ]
    assert reasoning_block.summary_text == "Draft summary"
    assert done_reasoning_block.summary_text == "Draft summary"


async def test_assemble_stream_message_done_text_overrides_accumulated_deltas() -> None:
    """MESSAGE_DONE text overrides accumulated deltas on the finalized text block."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(
                [
                    created_event("resp_refusal"),
                    message_added_event("msg_refusal"),
                    message_text_delta_event("No"),
                    message_done_event("msg_refusal", "No thanks"),
                    completed_event("stop"),
                ]
            ),
            source=_source(),
        )
    ]

    text_start = _expect_event_type(events[1], TextStartEvent)
    text_delta = _expect_event_type(events[2], TextDeltaEvent)
    text_end = _expect_event_type(events[3], TextEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)
    text_block = _expect_text_block(text_end.block)
    done_text_block = _expect_text_block(done.blocks[0])

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert text_start.content_index == 0
    assert text_delta.content_index == 0
    assert text_end.content_index == 0
    assert text_delta.delta == "No"
    assert text_block.text == "No thanks"
    assert done_text_block.text == "No thanks"


async def test_assemble_stream_maps_tool_call_events() -> None:
    """Accumulates tool-call events onto terminal stream blocks."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(_tool_call_events()),
            source=_source(),
        )
    ]

    _assert_tool_call_event_sequence(events)
    _assert_tool_call_stream_content(events)


@pytest.mark.parametrize(
    ("terminal_event", "expected_message"),
    [
        pytest.param(
            failed_event("Model overloaded"),
            "Model overloaded",
            id="failed",
        ),
        pytest.param(
            incomplete_event(
                "error",
                "OpenAI response was truncated by the content filter.",
            ),
            "OpenAI response was truncated by the content filter.",
            id="incomplete-error",
        ),
    ],
)
async def test_assemble_stream_maps_terminal_failures_into_error_events(
    terminal_event: NormalizedEvent,
    expected_message: str,
) -> None:
    """Builds an error stream event for failed and incomplete-error responses."""

    events = [
        event
        async for event in assemble_stream(
            async_stream([created_event("resp_error"), terminal_event]),
            source=_source(),
        )
    ]

    error = _expect_event_type(events[1], StreamErrorEvent)

    assert [event.type for event in events] == ["stream_start", "stream_error"]
    assert error.error_message == expected_message
    assert error.stop_reason == "error"
    assert error.response_id == "resp_error"


async def test_assemble_stream_maps_incomplete_length_into_done() -> None:
    """Builds a done event for non-error incomplete responses."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(
                [
                    created_event("resp_incomplete"),
                    message_added_event("msg_incomplete"),
                    message_text_delta_event("Partial answer"),
                    message_done_event("msg_incomplete", "Partial answer"),
                    incomplete_event("length", "OpenAI response incomplete."),
                ]
            ),
            source=_source(),
        )
    ]

    done = _expect_event_type(events[-1], StreamDoneEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert done.stop_reason == "length"
    assert _expect_text_block(done.blocks[0]).text == "Partial answer"


async def test_assemble_stream_stops_consuming_events_after_terminal_event() -> None:
    """Stops assembly once a terminal normalized event has been emitted."""

    events = [
        event
        async for event in assemble_stream(
            async_stream(
                [
                    created_event("resp_done"),
                    completed_event("stop"),
                    message_added_event("msg_after_done"),
                    message_text_delta_event("ignored"),
                ]
            ),
            source=_source(),
        )
    ]

    done = _expect_event_type(events[1], StreamDoneEvent)

    assert [event.type for event in events] == ["stream_start", "stream_done"]
    assert done.response_id == "resp_done"
    assert done.blocks == []


def _reasoning_text_events() -> list[NormalizedEvent]:
    """Build normalized events for a reasoning-plus-text response."""

    return [
        created_event("resp_success"),
        reasoning_added_event("rs_123"),
        reasoning_delta_event("Exploring "),
        reasoning_delta_event("reasoning traces"),
        reasoning_delta_event("\n\n"),
        reasoning_delta_event("Formulating "),
        reasoning_delta_event("reasoning traces"),
        reasoning_done_event(
            item_id="rs_123",
            summary_text=_combined_reasoning_summary(),
            reasoning_signature='{"id":"rs_123"}',
        ),
        message_added_event("msg_123"),
        message_text_delta_event("Hello"),
        message_text_delta_event(" world"),
        message_done_event("msg_123", "Hello world"),
        completed_event("stop"),
    ]


def _assert_reasoning_text_event_sequence(
    events: Sequence[ProviderStreamEvent],
) -> None:
    """Assert event order and content indexes for reasoning-plus-text output."""

    start = _expect_event_type(events[0], StreamStartEvent)
    assert [event.type for event in events] == [
        "stream_start",
        "reasoning_start",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_delta",
        "reasoning_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert start.response_id == "resp_success"
    assert start.source == _source()
    _assert_reasoning_content_indexes(events)
    _assert_text_content_indexes(events)


def _assert_reasoning_text_stream_content(
    events: Sequence[ProviderStreamEvent],
) -> None:
    """Assert streamed reasoning and text deltas."""

    reasoning_deltas = [
        _expect_event_type(events[index], ReasoningDeltaEvent).delta
        for index in range(2, 7)
    ]
    assert reasoning_deltas == [
        "Exploring ",
        "reasoning traces",
        "\n\n",
        "Formulating ",
        "reasoning traces",
    ]
    assert _expect_event_type(events[9], TextDeltaEvent).delta == "Hello"
    assert _expect_event_type(events[10], TextDeltaEvent).delta == " world"


def _assert_reasoning_text_terminal_blocks(
    events: Sequence[ProviderStreamEvent],
) -> None:
    """Assert final reasoning/text blocks and replay metadata."""

    reasoning_end = _expect_event_type(events[7], ReasoningEndEvent)
    text_end = _expect_event_type(events[11], TextEndEvent)
    done = _expect_event_type(events[12], StreamDoneEvent)
    final_reasoning_block = _expect_reasoning_block(reasoning_end.block)
    done_reasoning_block = _expect_reasoning_block(done.blocks[0])

    assert final_reasoning_block.summary_text == _combined_reasoning_summary()
    assert _expect_metadata_string(final_reasoning_block, "reasoning_signature") == (
        '{"id":"rs_123"}'
    )
    assert _expect_text_block(text_end.block).text == "Hello world"
    assert done.response_id == "resp_success"
    assert done.source == _source()
    assert done_reasoning_block.summary_text == _combined_reasoning_summary()
    assert _expect_metadata_string(done_reasoning_block, "reasoning_signature") == (
        '{"id":"rs_123"}'
    )
    assert _expect_text_block(done.blocks[1]).text == "Hello world"


def _assert_reasoning_content_indexes(events: Sequence[ProviderStreamEvent]) -> None:
    """Assert content indexes for reasoning events."""

    assert _expect_event_type(events[1], ReasoningStartEvent).content_index == 0
    for index in range(2, 7):
        assert _expect_event_type(events[index], ReasoningDeltaEvent).content_index == 0
    assert _expect_event_type(events[7], ReasoningEndEvent).content_index == 0


def _assert_text_content_indexes(events: Sequence[ProviderStreamEvent]) -> None:
    """Assert content indexes for text events."""

    assert _expect_event_type(events[8], TextStartEvent).content_index == 1
    assert _expect_event_type(events[9], TextDeltaEvent).content_index == 1
    assert _expect_event_type(events[10], TextDeltaEvent).content_index == 1
    assert _expect_event_type(events[11], TextEndEvent).content_index == 1


def _tool_call_events() -> list[NormalizedEvent]:
    """Build normalized events for a tool-call response."""

    return [
        created_event("resp_tool_call"),
        tool_call_added_event(
            provider_item_id="fc_123",
            call_id="call_123",
            name="get_weather",
            arguments={},
        ),
        tool_call_arguments_delta_event('{"'),
        tool_call_arguments_delta_event('city":"Munich"}'),
        tool_call_arguments_done_event({"city": "Munich"}),
        tool_call_done_event(
            provider_item_id="fc_123",
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Munich"},
        ),
        completed_event("tool_use"),
    ]


def _assert_tool_call_event_sequence(events: Sequence[ProviderStreamEvent]) -> None:
    """Assert event order and content indexes for tool-call output."""

    assert [event.type for event in events] == [
        "stream_start",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
        "stream_done",
    ]
    assert _expect_event_type(events[1], ToolCallStartEvent).content_index == 0
    assert _expect_event_type(events[2], ToolCallDeltaEvent).content_index == 0
    assert _expect_event_type(events[3], ToolCallDeltaEvent).content_index == 0
    assert _expect_event_type(events[4], ToolCallEndEvent).content_index == 0


def _assert_tool_call_stream_content(events: Sequence[ProviderStreamEvent]) -> None:
    """Assert streamed tool-call deltas and final blocks."""

    tool_call_delta_one = _expect_event_type(events[2], ToolCallDeltaEvent)
    tool_call_delta_two = _expect_event_type(events[3], ToolCallDeltaEvent)
    tool_call_end = _expect_event_type(events[4], ToolCallEndEvent)
    done = _expect_event_type(events[5], StreamDoneEvent)
    tool_call_block = _expect_tool_call_block(tool_call_end.block)

    assert tool_call_delta_one.delta == '{"'
    assert tool_call_delta_two.delta == 'city":"Munich"}'
    assert tool_call_block.call_id == "call_123"
    assert tool_call_block.name == "get_weather"
    assert _expect_metadata_string(tool_call_block, "provider_item_id") == "fc_123"
    assert tool_call_block.arguments == {"city": "Munich"}
    assert done.stop_reason == "tool_use"
    assert _expect_tool_call_block(done.blocks[0]).arguments == {"city": "Munich"}


def _combined_reasoning_summary() -> str:
    """Return the reasoning summary accumulated by the combined stream."""

    return "Exploring reasoning traces\n\nFormulating reasoning traces"


def _source() -> ProviderSource:
    """Build a deterministic provider source for assembler tests."""

    return ProviderSource(provider="openai", model="gpt-5.4")

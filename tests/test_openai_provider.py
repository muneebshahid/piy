"""Tests for OpenAI provider stream integration.

These tests document the first half of the streaming lifecycle:

1. Raw OpenAI SDK events are created in the test itself.
2. The provider passes those raw events through the SDK event adapter.
3. The adapter emits normalized events, and ``assemble_stream`` turns them into
   app-level ``StreamEvent`` models.

The focused adapter and assembler tests own the detailed event matrix. This file
keeps provider coverage at the transport wiring boundary.
"""

from typing import cast

from openai import AsyncOpenAI

from tile.providers.openai import OpenAIProvider
from tile.providers.openai.serialization import serialize_history_items, serialize_tools
from tile.types.conversation import UserMessage
from tile.types.stream_events import StreamStartEvent
from tile.types.tools import ToolDefinition
from tests.support.openai_response_events import (
    FakeOpenAIClient,
    build_fake_openai_client,
    message_added_event,
    message_done_event,
    response_completed_event,
    response_created_event,
    text_delta_event,
)
from tests.support.stream_assertions import expect_stream_event as _expect_event_type
from tests.support.tool_definitions import city_text_fn, city_tool


def _provider(client: FakeOpenAIClient) -> OpenAIProvider:
    """Build a configured OpenAI provider around a fake SDK client."""

    return OpenAIProvider(
        client=cast("AsyncOpenAI", client),
        model="gpt-5.4",
        reasoning={"effort": "medium"},
    )


def _sample_tools() -> list[ToolDefinition]:
    """Build a single deterministic weather tool definition."""

    return [
        city_tool(
            "get_weather",
            "Return a simple weather report for a city.",
            city_text_fn,
        )
    ]


async def test_stream_maps_raw_events_into_text_stream() -> None:
    """Pass raw SDK events through the provider stream pipeline."""

    raw_events = [
        response_created_event(1, "resp_success"),
        message_added_event(2, "msg_123", output_index=0),
        text_delta_event(4, "msg_123", "Hello", output_index=0),
        message_done_event(
            5,
            "msg_123",
            [{"type": "output_text", "text": "Hello", "annotations": []}],
            output_index=0,
        ),
        response_completed_event(6, "resp_success"),
    ]
    client = build_fake_openai_client(raw_events)

    event_stream = await _provider(client).stream(
        history=[UserMessage(content="hello")],
        instructions="Follow the repo conventions.",
        tools=None,
    )
    events = [event async for event in event_stream]

    start = _expect_event_type(events[0], StreamStartEvent)

    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_done",
    ]
    assert start.response_id == "resp_success"
    assert start.source.provider == "openai"
    assert start.source.model == "gpt-5.4"
    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
    )


async def test_stream_passes_serialized_tools_when_provided() -> None:
    """Forward serialized tool definitions to the SDK create call."""

    client = build_fake_openai_client([response_completed_event(1, "resp_tools")])
    tools = _sample_tools()

    event_stream = await _provider(client).stream(
        history=[UserMessage(content="hello")],
        instructions="Follow the repo conventions.",
        tools=tools,
    )
    _ = [event async for event in event_stream]

    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
        tools=serialize_tools(tools),
    )


async def test_stream_closes_the_sdk_transport_when_the_stream_ends() -> None:
    """Forward closure through the adapter chain down to the SDK stream."""

    raw_events = [
        response_created_event(1, "resp_close"),
        message_added_event(2, "msg_close", output_index=0),
        message_done_event(
            3,
            "msg_close",
            [{"type": "output_text", "text": "Hello", "annotations": []}],
            output_index=0,
        ),
        response_completed_event(4, "resp_close"),
    ]
    client = build_fake_openai_client(raw_events)

    event_stream = await _provider(client).stream(
        history=[UserMessage(content="hello")],
        instructions="Follow the repo conventions.",
        tools=None,
    )
    _ = [event async for event in event_stream]

    assert client.responses.create.return_value.closed

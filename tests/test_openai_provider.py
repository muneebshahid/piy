"""Tests for OpenAI provider stream integration.

These tests document the first half of the streaming lifecycle:

1. Raw OpenAI SDK events are created in the test itself.
2. The provider passes those raw events through the SDK event adapter.
3. The adapter emits normalized events, and ``assemble_stream`` turns them into
   app-level ``StreamEvent`` models.

The focused adapter and assembler tests own the detailed event matrix. This file
keeps provider coverage at the transport wiring boundary.
"""

import asyncio
from collections.abc import Sequence
from typing import cast

from openai import AsyncOpenAI

from tile.providers.openai import OpenAIProvider
from tile.providers.openai.serialization import serialize_history_items
from tile.types.conversation import UserMessage
from tile.types.stream_events import (
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from tile.types.tools import ToolDefinition, ToolResult
from tests.support.openai_response_events import (
    FakeOpenAIClient,
    build_fake_openai_client,
    message_added_event,
    message_done_event,
    response_completed_event,
    response_created_event,
    text_delta_event,
)
from tests.support.stream_assertions import (
    expect_stream_event as _expect_event_type,
    expect_text_block as _expect_text_block,
)
from tests.support.tool_definitions import CityInput, city_tool


def _collect_events(
    client: FakeOpenAIClient,
    tools: Sequence[ToolDefinition] | None = None,
) -> list[ProviderStreamEvent]:
    async def _collect() -> list[ProviderStreamEvent]:
        provider = OpenAIProvider(
            client=cast("AsyncOpenAI", client),
            model="gpt-5.4",
            reasoning={"effort": "medium"},
        )
        event_stream = await provider.stream(
            history=[UserMessage(content="hello")],
            instructions="Follow the repo conventions.",
            tools=tools,
        )
        return [event async for event in event_stream]

    return asyncio.run(_collect())


def _sample_tools() -> list[ToolDefinition]:
    return [
        city_tool(
            "get_weather",
            "Return a simple weather report for a city.",
            _sample_tool_fn,
        )
    ]


async def _sample_tool_fn(params: CityInput) -> ToolResult:
    """Return a deterministic payload for provider-only tool definitions."""

    return ToolResult.text(f"city={params.city}")


def test_stream_maps_raw_events_into_text_stream() -> None:
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
    events = _collect_events(client)

    start = _expect_event_type(events[0], StreamStartEvent)
    text_start = _expect_event_type(events[1], TextStartEvent)
    text_delta = _expect_event_type(events[2], TextDeltaEvent)
    text_end = _expect_event_type(events[3], TextEndEvent)
    done = _expect_event_type(events[4], StreamDoneEvent)

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
    assert text_start.content_index == 0
    assert text_delta.content_index == 0
    assert text_delta.delta == "Hello"
    assert text_end.content_index == 0
    assert _expect_text_block(text_end.block).text == "Hello"
    assert done.response_id == "resp_success"
    assert _expect_text_block(done.blocks[0]).text == "Hello"
    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
    )


def test_stream_passes_serialized_tools_when_provided() -> None:
    client = build_fake_openai_client([response_completed_event(1, "resp_tools")])
    tools = _sample_tools()

    _collect_events(client, tools=tools)

    client.responses.create.assert_awaited_once_with(
        model="gpt-5.4",
        input=serialize_history_items([UserMessage(content="hello")]),
        instructions="Follow the repo conventions.",
        stream=True,
        reasoning={"effort": "medium"},
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Return a simple weather report for a city.",
                "parameters": tools[0].input_schema,
                "strict": False,
                "defer_loading": False,
            }
        ],
    )


def test_stream_closes_the_sdk_transport_when_the_stream_ends() -> None:
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

    _collect_events(client)

    assert client.responses.create.return_value.closed

"""Provider stream helpers for stateless agent tests."""

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock

from tile.providers.base import Provider
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.stream_events import (
    AssistantBlock,
    ProviderMetadata,
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    StopReason,
    TextBlock,
    ToolCallBlock,
)
from tile.types.tools import JsonObject, ToolDefinition
from tests.support.async_streams import async_stream

TEST_PROVIDER = "test"


class ProviderStreamMock(Provider):
    """Async mock-backed fake provider stream with typed call inspectors."""

    def __init__(
        self,
        streams: Sequence[Sequence[ProviderStreamEvent]],
        *,
        model: str = "gpt-5.4",
        reasoning: JsonObject | None = None,
    ) -> None:
        """Create a fake provider stream from queued event streams."""

        super().__init__(model=model, reasoning=reasoning)
        self.mock = AsyncMock(side_effect=[async_stream(stream) for stream in streams])

    @property
    def name(self) -> str:
        """Return the deterministic test provider identity."""

        return TEST_PROVIDER

    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return the next configured provider event stream."""

        return await self.mock(
            tuple(history),
            instructions=instructions,
            tools=tools,
        )

    @property
    def await_count(self) -> int:
        """Return how many provider calls were awaited."""

        return self.mock.await_count

    def assert_awaited_once(self) -> None:
        """Assert the provider stream was awaited exactly once."""

        self.mock.assert_awaited_once()

    def instructions(self) -> str:
        """Return the instructions from the only provider call."""

        self.assert_awaited_once()
        await_args = self.mock.await_args
        assert await_args is not None
        instructions = await_args.kwargs["instructions"]
        assert isinstance(instructions, str)
        return instructions

    def history(self, index: int) -> tuple[ConversationItem, ...]:
        """Return model-visible history from one provider call."""

        await_args = self.mock.await_args_list[index]
        history = await_args.args[0]
        assert isinstance(history, tuple)
        return history

    def tools(self, index: int) -> tuple[ToolDefinition, ...] | None:
        """Return tool definitions from one provider call."""

        await_args = self.mock.await_args_list[index]
        tools = await_args.kwargs["tools"]
        assert tools is None or isinstance(tools, tuple)
        return tools


class GatedProviderStreamMock(ProviderStreamMock):
    """Provider stream fake that blocks each response on a release event."""

    def __init__(
        self,
        releases: Sequence[asyncio.Event],
        *,
        model: str = "gpt-5.4",
    ) -> None:
        """Create a gated provider stream fake with one release per call."""

        Provider.__init__(self, model=model)
        self._releases = tuple(releases)
        self._next_stream_index = 0
        self.mock = AsyncMock(side_effect=self._stream)

    async def _stream(
        self,
        _history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return the next provider stream after its release event is set."""

        _ = instructions, tools
        index = self._next_stream_index
        self._next_stream_index += 1
        await self._releases[index].wait()
        return async_stream(final_text_stream(f"resp_{index}", f"answer {index}"))


class GatedQueuedProviderStreamMock(ProviderStreamMock):
    """Queued provider streams whose individual calls require release."""

    def __init__(
        self,
        streams: Sequence[Sequence[ProviderStreamEvent]],
        releases: Sequence[asyncio.Event],
        *,
        model: str = "gpt-5.4",
    ) -> None:
        """Create one gate for each configured provider stream."""

        if len(streams) != len(releases):
            raise ValueError("Each queued stream requires one release event.")
        Provider.__init__(self, model=model)
        self._streams = tuple(tuple(stream) for stream in streams)
        self._releases = tuple(releases)
        self._next_stream_index = 0
        self.mock = AsyncMock(side_effect=self._stream)

    async def _stream(
        self,
        _history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return the next queued stream after its release event is set."""

        _ = instructions, tools
        index = self._next_stream_index
        self._next_stream_index += 1
        await self._releases[index].wait()
        return async_stream(self._streams[index])


def final_text_stream(response_id: str, text: str) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that completes with final text."""

    return [
        stream_start(response_id),
        stream_done(response_id, blocks=[TextBlock(text=text)]),
    ]


def empty_stream(response_id: str) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that completes without content blocks."""

    return [
        stream_start(response_id),
        stream_done(response_id),
    ]


def error_stream(response_id: str, error_message: str) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that fails before completion."""

    return [
        stream_start(response_id),
        stream_error(response_id, error_message),
    ]


def tool_call_stream(
    *,
    response_id: str,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
    provider_item_id: str | None = None,
) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that requests one tool call."""

    return [
        stream_start(response_id),
        stream_done(
            response_id,
            stop_reason="tool_use",
            blocks=[
                tool_call_block(
                    call_id=call_id,
                    name=tool_name,
                    arguments=arguments,
                    provider_item_id=provider_item_id,
                )
            ],
        ),
    ]


def stream_start(response_id: str) -> StreamStartEvent:
    """Build a deterministic provider stream start event."""

    return StreamStartEvent(source=provider_source(), response_id=response_id)


def stream_done(
    response_id: str,
    *,
    stop_reason: StopReason = "stop",
    blocks: Sequence[AssistantBlock] = (),
) -> StreamDoneEvent:
    """Build a deterministic provider stream completion event."""

    return StreamDoneEvent(
        source=provider_source(),
        response_id=response_id,
        stop_reason=stop_reason,
        blocks=list(blocks),
    )


def stream_error(
    response_id: str,
    error_message: str,
    *,
    blocks: Sequence[AssistantBlock] = (),
) -> StreamErrorEvent:
    """Build a deterministic provider stream error event."""

    return StreamErrorEvent(
        source=provider_source(),
        response_id=response_id,
        error_message=error_message,
        blocks=list(blocks),
    )


def tool_call_block(
    *,
    call_id: str,
    name: str,
    arguments: JsonObject,
    provider_item_id: str | None = None,
) -> ToolCallBlock:
    """Build a tool-call block with optional provider replay metadata."""

    return ToolCallBlock(
        call_id=call_id,
        name=name,
        arguments=arguments,
        provider_metadata=ProviderMetadata.from_values(
            provider_item_id=provider_item_id
        ),
    )


def provider_source() -> ProviderSource:
    """Build the deterministic provider source used by agent tests."""

    return ProviderSource(provider=TEST_PROVIDER, model="gpt-5.4")

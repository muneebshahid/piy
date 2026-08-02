"""OpenAI Responses API provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from tile.providers.base import Provider
from tile.providers.openai.serialization import serialize_history_items, serialize_tools
from tile.providers.openai.sdk_event_adapter import normalize_sdk_events
from tile.providers.openai.stream_assembler import assemble_stream
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderSource
from tile.types.tools import JsonObject, ToolDefinition

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


class Reasoning(TypedDict, total=False):
    """OpenAI reasoning options bound to a provider at creation."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    summary: Literal["auto", "concise", "detailed"]


class OpenAIProvider(Provider):
    """Configured OpenAI Responses API provider."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        reasoning: Reasoning | None = None,
    ) -> None:
        """Bind an OpenAI client, model, and reasoning configuration."""

        super().__init__(
            model=model,
            reasoning=cast("JsonObject | None", reasoning),
        )
        self._client = client

    @property
    def name(self) -> str:
        """Return the OpenAI provider identity."""

        return "openai"

    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> AsyncEventStream:
        """Stream assistant events through the OpenAI SDK transport."""

        request_params = _build_stream_request_params(
            history,
            self.model,
            instructions=instructions,
            reasoning=cast("Reasoning | None", self.reasoning),
            tools=tools,
        )
        raw_stream = await self._client.responses.create(**request_params)
        return assemble_stream(
            normalize_sdk_events(_sdk_stream_events(raw_stream)),
            source=ProviderSource(provider=self.name, model=self.model),
        )


async def _sdk_stream_events(
    raw_stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """Yield SDK events, closing the transport stream on every exit.

    The SDK stream owns a live HTTP response; wrapping it in a generator
    lets the adapter chain above forward closure down to the transport.
    """

    try:
        async for event in raw_stream:
            yield event
    finally:
        await raw_stream.close()


def _build_stream_request_params(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: Reasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> ResponseCreateParamsStreaming:
    """Build the shared Responses API request payload for stream transports."""

    request_params: ResponseCreateParamsStreaming = {
        "model": model,
        "input": serialize_history_items(history),
        "reasoning": cast("OpenAIReasoning | None", reasoning),
        "instructions": instructions,
        "stream": True,
    }
    if tools:
        request_params["tools"] = serialize_tools(tools)
    return request_params

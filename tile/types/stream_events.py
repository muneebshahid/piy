"""Provider stream events and replayable assistant block contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field

from tile.types.tools import JsonObject

type StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class ProviderSource(BaseModel):
    """Provider and model that produced an assistant turn."""

    provider: str
    model: str | None = None


class ProviderMetadata(BaseModel):
    """Opaque provider-owned metadata needed for compatible replay."""

    data: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_values(cls, **values: str | None) -> Self | None:
        """Build metadata from provider values, omitting absent entries."""

        data: JsonObject = {
            key: value for key, value in values.items() if value is not None
        }
        if not data:
            return None
        return cls(data=data)

    def string_value(self, key: str) -> str | None:
        """Return a string metadata value when present."""

        value = self.data.get(key)
        if isinstance(value, str):
            return value
        return None


class AssistantBlockBase(BaseModel):
    """Base contract for replayable assistant content blocks."""

    provider_metadata: ProviderMetadata | None = None

    def metadata_string(self, key: str) -> str | None:
        """Return a provider metadata string value when present."""

        if self.provider_metadata is None:
            return None
        return self.provider_metadata.string_value(key)


class TextBlock(AssistantBlockBase):
    """Replayable assistant text block."""

    type: Literal["text"] = "text"
    text: str


class ReasoningBlock(AssistantBlockBase):
    """Replayable assistant reasoning block."""

    type: Literal["reasoning"] = "reasoning"
    summary_text: str


class ToolCallBlock(AssistantBlockBase):
    """Replayable assistant tool-call block."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: JsonObject = Field(default_factory=dict)


type AssistantBlock = TextBlock | ReasoningBlock | ToolCallBlock


class StreamEvent(BaseModel):
    """Base provider stream event."""

    type: str


class BlockStreamEvent(StreamEvent):
    """Provider stream event scoped to one assistant content block."""

    content_index: int


class StreamStartEvent(StreamEvent):
    """Marks creation of a provider response stream."""

    type: Literal["stream_start"] = "stream_start"
    source: ProviderSource
    response_id: str


class ReasoningStartEvent(BlockStreamEvent):
    """Marks the start of a reasoning block."""

    type: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDeltaEvent(BlockStreamEvent):
    """Carries incremental reasoning text for a block."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


class ReasoningEndEvent(BlockStreamEvent):
    """Carries a completed reasoning block."""

    type: Literal["reasoning_end"] = "reasoning_end"
    block: ReasoningBlock


class TextStartEvent(BlockStreamEvent):
    """Marks the start of a text block."""

    type: Literal["text_start"] = "text_start"


class TextDeltaEvent(BlockStreamEvent):
    """Carries incremental text for a block."""

    type: Literal["text_delta"] = "text_delta"
    delta: str


class TextEndEvent(BlockStreamEvent):
    """Carries a completed text block."""

    type: Literal["text_end"] = "text_end"
    block: TextBlock


class ToolCallStartEvent(BlockStreamEvent):
    """Marks the start of a tool-call block."""

    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    name: str


class ToolCallDeltaEvent(BlockStreamEvent):
    """Carries incremental tool-call argument JSON."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    delta: str


class ToolCallEndEvent(BlockStreamEvent):
    """Carries a completed tool-call block."""

    type: Literal["tool_call_end"] = "tool_call_end"
    block: ToolCallBlock


type StreamBlockEvent = ReasoningEndEvent | TextEndEvent | ToolCallEndEvent

type StreamUpdateEvent = (
    ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
)


class StreamDoneEvent(StreamEvent):
    """Marks successful provider stream completion."""

    type: Literal["stream_done"] = "stream_done"
    source: ProviderSource
    response_id: str | None = None
    stop_reason: StopReason
    blocks: list[AssistantBlock] = Field(default_factory=list)


class StreamErrorEvent(StreamEvent):
    """Marks failed provider stream completion."""

    type: Literal["stream_error"] = "stream_error"
    source: ProviderSource
    response_id: str | None = None
    stop_reason: StopReason = "error"
    error_message: str
    blocks: list[AssistantBlock] = Field(default_factory=list)


type StreamTerminalEvent = StreamDoneEvent | StreamErrorEvent
type ProviderStreamEvent = (
    StreamStartEvent | StreamUpdateEvent | StreamDoneEvent | StreamErrorEvent
)

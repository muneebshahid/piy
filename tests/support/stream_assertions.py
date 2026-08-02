"""Shared assertions for provider stream event tests."""

from tile.types.stream_events import (
    AssistantBlock,
    ReasoningBlock,
    StreamEvent,
    TextBlock,
    ToolCallBlock,
)


def expect_stream_event[TStreamEvent: StreamEvent](
    event: StreamEvent,
    event_type: type[TStreamEvent],
) -> TStreamEvent:
    """Assert and return a stream event with a precise runtime type."""

    assert isinstance(event, event_type)
    return event


def expect_reasoning_block(block: AssistantBlock) -> ReasoningBlock:
    """Assert and return a reasoning assistant block."""

    assert isinstance(block, ReasoningBlock)
    return block


def expect_text_block(block: AssistantBlock) -> TextBlock:
    """Assert and return a text assistant block."""

    assert isinstance(block, TextBlock)
    return block


def expect_tool_call_block(block: AssistantBlock) -> ToolCallBlock:
    """Assert and return a tool-call assistant block."""

    assert isinstance(block, ToolCallBlock)
    return block


def expect_metadata_string(
    block: AssistantBlock,
    key: str,
) -> str:
    """Assert that a block contains a provider metadata string value."""

    value = block.metadata_string(key)
    assert value is not None
    return value

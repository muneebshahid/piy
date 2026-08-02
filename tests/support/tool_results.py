"""Shared helpers for inspecting tool result content in tests."""

from typing import TypeVar

from tile.types.conversation import ToolResultTurn
from tile.types.tools import ToolResult, ToolTextContent

TDetails = TypeVar("TDetails")


def details_of(result: ToolResult, details_type: type[TDetails]) -> TDetails:
    """Return the typed details payload from a tool result."""

    details = result.details
    assert isinstance(details, details_type)
    return details


def tool_text(result: ToolResult | ToolResultTurn) -> str:
    """Return the single text block from a result or replay projection."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text

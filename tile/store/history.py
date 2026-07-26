"""Domain validation for replayable flat conversation history."""

from collections.abc import Sequence

from tile.store.base import InvalidHistoryError
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import ToolCallBlock


def validate_replayable_history(items: Sequence[ConversationItem]) -> None:
    """Reject a sequence that cannot be replayed safely to a provider."""

    if not items or not isinstance(items[0], UserMessage):
        raise InvalidHistoryError("Committed history must begin with a user message.")

    pending_calls: set[str] = set()
    seen_calls: set[str] = set()
    for item in items:
        if isinstance(item, UserMessage):
            _reject_pending_calls(pending_calls, boundary="a user message")
        elif isinstance(item, AssistantTurn):
            _reject_pending_calls(pending_calls, boundary="an assistant turn")
            if item.status != "completed":
                raise InvalidHistoryError(
                    "Committed assistant turns must be completed."
                )
            _register_tool_calls(item, pending_calls, seen_calls)
        elif isinstance(item, ToolResultTurn):
            _consume_tool_result(item, pending_calls)
    _reject_pending_calls(pending_calls, boundary="the end of history")


def _register_tool_calls(
    turn: AssistantTurn,
    pending_calls: set[str],
    seen_calls: set[str],
) -> None:
    """Register unique tool calls emitted by one completed assistant turn."""

    for block in turn.blocks:
        if not isinstance(block, ToolCallBlock):
            continue
        if block.call_id in seen_calls:
            raise InvalidHistoryError(f"Duplicate tool call id: {block.call_id}")
        seen_calls.add(block.call_id)
        pending_calls.add(block.call_id)


def _consume_tool_result(
    result: ToolResultTurn,
    pending_calls: set[str],
) -> None:
    """Match a tool result to one preceding unanswered tool call."""

    if result.call_id not in pending_calls:
        raise InvalidHistoryError(f"Tool result has no pending call: {result.call_id}")
    pending_calls.remove(result.call_id)


def _reject_pending_calls(
    pending_calls: set[str],
    *,
    boundary: str,
) -> None:
    """Require all prior tool calls to be answered before a boundary."""

    if not pending_calls:
        return
    call_ids = ", ".join(sorted(pending_calls))
    raise InvalidHistoryError(f"Unanswered tool calls before {boundary}: {call_ids}")

"""Run-local conversation buffering and replay healing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tile.events import (
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    ToolExecutionEndEvent,
)
from tile.store import HistoryItem
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import ToolCallBlock
from tile.types.tools import ToolTextContent


@dataclass
class _RunHistory:
    """Own provider-visible working history and the provisional run delta."""

    working: list[ConversationItem]
    delta: list[ConversationItem]

    @classmethod
    def start(
        cls,
        committed: Sequence[HistoryItem],
        *,
        prompt: str,
    ) -> _RunHistory:
        """Build a run-local buffer from committed history and its prompt."""

        delta: list[ConversationItem] = [UserMessage(content=prompt)]
        working = [
            *(envelope.item.model_copy(deep=True) for envelope in committed),
            *(item.model_copy(deep=True) for item in delta),
        ]
        return cls(working=working, delta=delta)

    def observe(self, event: AgentEvent) -> None:
        """Project one replayable event payload into both local histories."""

        item = _conversation_item_for(event)
        if item is None:
            return
        local_item = item.model_copy(deep=True)
        self.delta.append(local_item)
        self.working.append(local_item)

    def snapshot(self) -> tuple[ConversationItem, ...]:
        """Return defensive copies of the provisional run delta."""

        return tuple(item.model_copy(deep=True) for item in self.delta)

    def heal(self) -> None:
        """Append error results for tool calls interrupted before completion."""

        results = [
            ToolResultTurn(
                call_id=call.call_id,
                tool_name=call.name,
                content=[ToolTextContent(text="Tool execution did not complete.")],
                is_error=True,
            )
            for call in _unanswered_tool_calls(self.delta)
        ]
        self.delta.extend(results)
        self.working.extend(results)


def _conversation_item_for(event: AgentEvent) -> ConversationItem | None:
    """Return the replayable conversation item carried by an event."""

    if isinstance(event, MessageEndEvent):
        if event.assistant_turn.status != "completed":
            return None
        return event.assistant_turn
    if isinstance(event, ToolExecutionEndEvent):
        return event.outcome.tool_result_turn
    if isinstance(event, ResultFollowUpEvent):
        return event.message
    return None


def _unanswered_tool_calls(
    items: Sequence[ConversationItem],
) -> list[ToolCallBlock]:
    """Return assistant tool calls without a corresponding result."""

    answered = {item.call_id for item in items if isinstance(item, ToolResultTurn)}
    return [
        block
        for item in items
        if isinstance(item, AssistantTurn)
        for block in item.blocks
        if isinstance(block, ToolCallBlock) and block.call_id not in answered
    ]

"""Helpers for capturing stateless agent runs in tests."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from tile.agent import AgentResult, run_agent
from tile.events import AgentEvent, StreamFn
from tile.tool_executor import ToolExecutor
from tile.types.conversation import ConversationItem
from tile.types.tools import ToolDefinition


@dataclass(frozen=True)
class AgentRunCapture:
    """Events and terminal result captured from one stateless agent run."""

    events: list[AgentEvent]
    result: AgentResult


def collect_agent_run(
    history: Sequence[ConversationItem],
    *,
    stream_fn: StreamFn,
    model: str = "gpt-5.4",
    tools: Sequence[ToolDefinition] = (),
    instructions: str = "Base prompt.",
) -> AgentRunCapture:
    """Run an agent and capture its emitted events and terminal result."""

    async def _collect() -> AgentRunCapture:
        """Capture one run through the agent's emit and return boundaries."""

        events: list[AgentEvent] = []
        result = await run_agent(
            history,
            emit=events.append,
            stream_fn=stream_fn,
            model=model,
            tool_executor=ToolExecutor(tools),
            instructions=instructions,
        )
        return AgentRunCapture(events=events, result=result)

    return asyncio.run(_collect())


def collect_run_events(
    history: Sequence[ConversationItem],
    *,
    stream_fn: StreamFn,
    model: str = "gpt-5.4",
    tools: Sequence[ToolDefinition] = (),
    instructions: str = "Base prompt.",
) -> list[AgentEvent]:
    """Run an agent and return its emitted events."""

    return collect_agent_run(
        history,
        stream_fn=stream_fn,
        model=model,
        tools=tools,
        instructions=instructions,
    ).events

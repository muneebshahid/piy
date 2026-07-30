"""Stateless agent run loop that emits events and returns its result."""

from collections.abc import Sequence
from contextlib import aclosing
from dataclasses import dataclass

from tile.exceptions import (
    ProviderStreamProtocolError,
    TurnFailedError,
)
from tile.events import (
    AgentEndEvent,
    AgentStartEvent,
    EmitFn,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tile.tool_executor import ToolExecutor
from tile.types.conversation import AssistantTurn, ConversationItem
from tile.types.stream_events import (
    AssistantBlock,
    ProviderStreamEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from tile.types.tool_execution import ToolExecutionOutcome

ASSISTANT_MESSAGE_UPDATE_EVENT_TYPES = (
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)
"""Provider event types forwarded as assistant message updates."""


@dataclass(frozen=True)
class AgentResult:
    """Terminal facts produced by one successful stateless agent invocation.

    ``last_turn`` is the final completed assistant turn. ``tool_executions``
    contains every tool outcome produced across the invocation's provider
    turns.
    """

    last_turn: AssistantTurn
    tool_executions: tuple[ToolExecutionOutcome, ...]


@dataclass(frozen=True)
class _TurnResult:
    """Final assistant turn and tool executions produced by one provider stream."""

    assistant_turn: AssistantTurn
    tool_executions: tuple[ToolExecutionOutcome, ...]

    @property
    def conversation_items(self) -> tuple[ConversationItem, ...]:
        """Return the assistant turn followed by its tool result turns."""

        return (
            self.assistant_turn,
            *(execution.tool_result_turn for execution in self.tool_executions),
        )

    @property
    def should_continue(self) -> bool:
        """Return whether this turn requires another provider turn."""

        return bool(self.tool_executions) and not any(
            execution.terminate for execution in self.tool_executions
        )


async def run_agent(
    history: Sequence[ConversationItem],
    *,
    emit: EmitFn,
    stream_fn: StreamFn,
    model: str,
    tool_executor: ToolExecutor,
    instructions: str,
) -> AgentResult:
    """Run one stateless agent invocation and return its terminal facts.

    A successful tool result with ``terminate=True`` ends the loop after the
    current tool batch without another provider call. ``instructions`` is the
    complete system prompt, sent to the provider verbatim; the caller owns
    its composition.
    """

    emit(AgentStartEvent())
    result = await _run_agent_loop(
        run_history=list(history),
        emit=emit,
        stream_fn=stream_fn,
        model=model,
        instructions=instructions,
        tool_executor=tool_executor,
    )
    emit(AgentEndEvent())
    return result


async def _run_agent_loop(
    *,
    run_history: list[ConversationItem],
    emit: EmitFn,
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
) -> AgentResult:
    """Call the provider until a turn terminates or stops requesting tools."""

    tool_executions: list[ToolExecutionOutcome] = []
    while True:
        result = await _run_provider_turn(
            run_history=run_history,
            emit=emit,
            stream_fn=stream_fn,
            model=model,
            instructions=instructions,
            tool_executor=tool_executor,
        )
        run_history.extend(result.conversation_items)
        tool_executions.extend(result.tool_executions)
        if not result.should_continue:
            return AgentResult(
                last_turn=result.assistant_turn,
                tool_executions=tuple(tool_executions),
            )


async def _run_provider_turn(
    *,
    run_history: Sequence[ConversationItem],
    emit: EmitFn,
    stream_fn: StreamFn,
    model: str,
    instructions: str,
    tool_executor: ToolExecutor,
) -> _TurnResult:
    """Run one provider stream and return its terminal turn facts.

    The provider stream is closed on every exit so the transport does not
    depend on garbage collection for cleanup.
    """

    stream = await stream_fn(
        tuple(run_history),
        model,
        instructions=instructions,
        tools=tool_executor.tools,
    )
    async with aclosing(stream):
        async for event in stream:
            result = await _handle_stream_event(
                event,
                emit=emit,
                tool_executor=tool_executor,
            )
            if result is not None:
                return result
    raise ProviderStreamProtocolError(
        "Provider stream ended without StreamDoneEvent or StreamErrorEvent."
    )


async def _handle_stream_event(
    event: ProviderStreamEvent,
    *,
    emit: EmitFn,
    tool_executor: ToolExecutor,
) -> _TurnResult | None:
    """Route one provider stream event and return terminal turn facts."""

    match event:
        case StreamStartEvent():
            _emit_turn_start(event, emit=emit)
        case StreamDoneEvent():
            return await _handle_stream_done_event(
                event,
                emit=emit,
                tool_executor=tool_executor,
            )
        case StreamErrorEvent():
            _handle_stream_error_event(
                event,
                emit=emit,
            )
        case _ if isinstance(event, ASSISTANT_MESSAGE_UPDATE_EVENT_TYPES):
            emit(MessageUpdateEvent(stream_event=event))
    return None


def _emit_turn_start(event: StreamStartEvent, *, emit: EmitFn) -> None:
    """Emit the opening lifecycle events for one provider turn."""

    emit(TurnStartEvent())
    emit(MessageStartEvent(response_id=event.response_id))


async def _handle_stream_done_event(
    event: StreamDoneEvent,
    *,
    emit: EmitFn,
    tool_executor: ToolExecutor,
) -> _TurnResult:
    """Finalize a completed assistant message and execute requested tools."""

    turn = AssistantTurn.from_stream_done(event)
    emit(MessageEndEvent(assistant_turn=turn))
    tool_executions = await _execute_tools(
        turn.blocks,
        emit=emit,
        tool_executor=tool_executor,
    )
    emit(
        TurnEndEvent(
            assistant_turn=turn,
            tool_executions=tuple(tool_executions),
        )
    )
    return _TurnResult(
        assistant_turn=turn,
        tool_executions=tuple(tool_executions),
    )


def _handle_stream_error_event(
    event: StreamErrorEvent,
    *,
    emit: EmitFn,
) -> None:
    """Finalize a failed assistant message."""

    turn = AssistantTurn.from_stream_error(event)
    emit(MessageEndEvent(assistant_turn=turn))
    raise TurnFailedError(turn=turn)


async def _execute_tools(
    blocks: Sequence[AssistantBlock],
    *,
    emit: EmitFn,
    tool_executor: ToolExecutor,
) -> list[ToolExecutionOutcome]:
    """Execute every tool call and return its outcome."""

    outcomes: list[ToolExecutionOutcome] = []
    for tool_call in _collect_tool_calls(blocks):
        outcome = await _execute_tool(
            tool_call,
            emit=emit,
            tool_executor=tool_executor,
        )
        outcomes.append(outcome)
    return outcomes


async def _execute_tool(
    tool_call: ToolCallBlock,
    *,
    emit: EmitFn,
    tool_executor: ToolExecutor,
) -> ToolExecutionOutcome:
    """Emit the full lifecycle for one tool execution and return its outcome."""

    emit(
        ToolExecutionStartEvent(
            call_id=tool_call.call_id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )
    )
    outcome = await tool_executor.execute(
        call_id=tool_call.call_id,
        tool_name=tool_call.name,
        arguments=tool_call.arguments,
    )
    emit(ToolExecutionEndEvent(outcome=outcome))
    return outcome


def _collect_tool_calls(blocks: Sequence[AssistantBlock]) -> list[ToolCallBlock]:
    """Collect tool calls from finalized assistant blocks."""

    return [
        block.model_copy(deep=True)
        for block in blocks
        if isinstance(block, ToolCallBlock)
    ]

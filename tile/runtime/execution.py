"""Prompt programs: what a prompt run emits and how it concludes.

Execution pushes inner events through the run's publish callable and
returns the ``RunOutcome``. It never publishes run lifecycle events and
never touches persistence — its dependency contract carries no Store, so
the boundary is structural: the run turns the returned outcome
— or the exception or cancellation that replaces it — into the terminal
run end event, so a duplicated or missing run end is unrepresentable
here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel

from tile.agent import run_agent
from tile.events import (
    AgentEvent,
    MessageEndEvent,
    ResultFollowUpEvent,
    StreamFn,
    ToolExecutionEndEvent,
)
from tile.prompt import build_system_prompt
from tile.result import (
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    AgentFailure,
    Completed,
    Failed,
    RunOutcome,
)
from tile.tool_executor import ToolExecutor
from tile.tools.complete import CompleteDetails
from tile.tools.complete import tool as complete_tool
from tile.tools.fail import FailDetails
from tile.tools.fail import tool as fail_tool
from tile.types.conversation import AssistantTurn, ConversationItem, UserMessage
from tile.types.stream_events import TextBlock
from tile.types.tools import ToolDetails

PublishFn = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class _ExecutionDependencies:
    """Caller-constructed dependencies a prompt program may touch.

    Deliberately excludes persistence. Execution receives one run-local
    history snapshot and drives only the provider and tools.
    """

    stream_fn: StreamFn
    model: str
    instructions: str
    cwd: Path
    auto_mode: bool
    tool_executor: ToolExecutor


class TurnFailedError(RuntimeError):
    """Raised when an agent run ends without a completed assistant turn."""

    def __init__(self, turn: AssistantTurn | None) -> None:
        """Preserve the failed turn while exposing a concise exception message."""

        self.turn = turn
        super().__init__(_turn_failure_message(turn))


async def execute_prompt(
    publish: PublishFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
    result: type[BaseModel] | None,
) -> RunOutcome:
    """Run one prompt program, publishing inner events, and return its outcome."""

    if result is None:
        return await _execute_plain(publish, deps=deps, history=history)
    return await _execute_typed(
        publish,
        deps=deps,
        history=history,
        result=result,
    )


async def _execute_plain(
    publish: PublishFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> RunOutcome:
    """Run one plain agent invocation and conclude with its text outcome."""

    observation = _AgentRunObservation()
    await _run_attempt(
        publish,
        observation,
        deps=deps,
        history=history,
    )
    turn = _require_completed_turn(observation.last_turn)
    return Completed(value=_assistant_text(turn))


async def _execute_typed(
    publish: PublishFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
    result: type[BaseModel],
) -> RunOutcome:
    """Run agent attempts until the required result is produced or exhausted."""

    attempt_dependencies = _typed_attempt_dependencies(deps, result)
    for attempt in range(MAX_RESULT_FOLLOW_UPS + 1):
        observation = _AgentRunObservation()
        await _run_attempt(
            publish,
            observation,
            deps=attempt_dependencies,
            history=history,
        )
        _require_completed_turn(observation.last_turn)
        outcome = _result_outcome(observation.terminal_details)
        if outcome is not None:
            return outcome
        if attempt < MAX_RESULT_FOLLOW_UPS:
            publish(ResultFollowUpEvent(message=UserMessage(content=RESULT_FOLLOW_UP)))
    return Failed(cause=AgentFailure(reason=NO_RESULT_REASON))


async def _run_attempt(
    publish: PublishFn,
    observation: _AgentRunObservation,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> None:
    """Drive one stateless agent attempt, publishing every event.

    The system prompt is composed here, per attempt, so project context
    and the environment lines stay current across attempts. The agent
    generator is closed on every exit: a publish failure leaves it
    suspended mid-yield, and its cleanup must not wait for garbage
    collection.
    """

    events = run_agent(
        history,
        stream_fn=deps.stream_fn,
        model=deps.model,
        tool_executor=deps.tool_executor,
        instructions=build_system_prompt(
            deps.instructions,
            deps.cwd,
            auto_mode=deps.auto_mode,
        ),
    )
    async with aclosing(events):
        async for event in events:
            observation.observe(event)
            publish(event)


def _typed_attempt_dependencies(
    deps: _ExecutionDependencies,
    result: type[BaseModel],
) -> _ExecutionDependencies:
    """Resolve one typed prompt's shared attempt dependencies."""

    return replace(
        deps,
        instructions=f"{deps.instructions}\n\n{RESULT_CONTRACT}",
        tool_executor=ToolExecutor(
            (*deps.tool_executor.tools, complete_tool(result), fail_tool)
        ),
    )


@dataclass
class _AgentRunObservation:
    """Result-relevant facts observed during one stateless agent run."""

    last_turn: AssistantTurn | None = None
    terminal_details: ToolDetails | None = None

    def observe(self, event: AgentEvent) -> None:
        """Record the latest assistant turn and first terminating tool details."""

        if isinstance(event, MessageEndEvent):
            self.last_turn = event.assistant_turn
        if (
            self.terminal_details is None
            and isinstance(event, ToolExecutionEndEvent)
            and event.outcome.terminate
            and isinstance(event.outcome.details, CompleteDetails | FailDetails)
        ):
            self.terminal_details = event.outcome.details


def _require_completed_turn(turn: AssistantTurn | None) -> AssistantTurn:
    """Return the run's final assistant turn, raising when it did not complete."""

    if turn is None:
        raise TurnFailedError(turn)
    if turn.status != "completed":
        raise TurnFailedError(turn)
    return turn


def _turn_failure_message(turn: AssistantTurn | None) -> str:
    """Return the public message for an unsuccessful assistant turn."""

    if turn is None:
        return "The agent run ended without an assistant turn."
    return turn.error_message or "The assistant turn failed."


def _result_outcome(terminal_details: ToolDetails | None) -> RunOutcome | None:
    """Build a terminal outcome, or return None when a result remains missing."""

    if isinstance(terminal_details, CompleteDetails):
        return Completed(value=terminal_details.value)
    if isinstance(terminal_details, FailDetails):
        return Failed(cause=AgentFailure(reason=terminal_details.reason))
    return None


def _assistant_text(turn: AssistantTurn) -> str:
    """Join one assistant turn's text blocks."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )

"""Prompt programs: what a prompt run emits and how it concludes.

Execution emits inner events through the run's emit callable and returns the
``RunOutcome``. It never emits run lifecycle events and never touches
persistence — its dependency contract carries no Store, so the boundary is
structural: the run turns the returned outcome, exception, or cancellation
into the terminal run end event.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from tile.agent import AgentResult, run_agent
from tile.events import EmitFn, ResultFollowUpEvent
from tile.providers.base import Provider
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
from tile.types.tool_execution import ToolExecutionOutcome


@dataclass(frozen=True)
class _ExecutionDependencies:
    provider: Provider
    system_prompt: str
    tool_executor: ToolExecutor
    result_type: type[BaseModel] | None


def build_execution_dependencies(
    *,
    provider: Provider,
    system_prompt: str,
    tool_executor: ToolExecutor,
    result_type: type[BaseModel] | None,
) -> _ExecutionDependencies:
    if result_type is not None:
        system_prompt = f"{system_prompt}\n\n{RESULT_CONTRACT}"
        tool_executor = _result_contract_tool_executor(tool_executor, result_type)
    return _ExecutionDependencies(
        provider=provider,
        system_prompt=system_prompt,
        tool_executor=tool_executor,
        result_type=result_type,
    )


async def execute_prompt(
    emit: EmitFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> RunOutcome:
    """Run one prompt program, emitting inner events, and return its outcome."""

    if deps.result_type is None:
        return await _execute_plain(emit, deps=deps, history=history)
    return await _execute_with_result_contract(
        emit,
        deps=deps,
        history=history,
    )


async def _execute_plain(
    emit: EmitFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> RunOutcome:
    """Run one plain agent invocation and conclude with its text outcome."""

    agent_result = await _run_attempt(
        emit,
        deps=deps,
        history=history,
    )
    return Completed(value=_assistant_text(agent_result.last_turn))


async def _execute_with_result_contract(
    emit: EmitFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> RunOutcome:
    """Run agent attempts until the required result is produced or exhausted."""

    for follow_ups_used in range(MAX_RESULT_FOLLOW_UPS + 1):
        agent_result = await _run_attempt(
            emit,
            deps=deps,
            history=history,
        )
        outcome = _result_outcome(agent_result.tool_executions)
        if outcome is not None:
            return outcome
        if follow_ups_used < MAX_RESULT_FOLLOW_UPS:
            emit(ResultFollowUpEvent(message=UserMessage(content=RESULT_FOLLOW_UP)))
    return Failed(cause=AgentFailure(reason=NO_RESULT_REASON))


async def _run_attempt(
    emit: EmitFn,
    *,
    deps: _ExecutionDependencies,
    history: Sequence[ConversationItem],
) -> AgentResult:
    return await run_agent(
        history,
        emit=emit,
        provider=deps.provider,
        tool_executor=deps.tool_executor,
        instructions=deps.system_prompt,
    )


def _result_contract_tool_executor(
    tool_executor: ToolExecutor,
    result_type: type[BaseModel],
) -> ToolExecutor:
    return ToolExecutor((*tool_executor.tools, complete_tool(result_type), fail_tool))


def _result_outcome(
    tool_executions: Sequence[ToolExecutionOutcome],
) -> RunOutcome | None:
    """Build a terminal outcome, or return None when a result remains missing."""

    for execution in tool_executions:
        match execution.details:
            case CompleteDetails(value=value):
                return Completed(value=value)
            case FailDetails(reason=reason):
                return Failed(cause=AgentFailure(reason=reason))
    return None


def _assistant_text(turn: AssistantTurn) -> str:
    """Join one assistant turn's text blocks."""

    return "\n\n".join(
        block.text for block in turn.blocks if isinstance(block, TextBlock)
    )

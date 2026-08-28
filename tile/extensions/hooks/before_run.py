from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
)
from tile.types.stream_events import ToolCallBlock


class BeforeRunContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    run_id: str
    system_prompt: str
    messages: tuple[ConversationItem, ...]


class BeforeRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = None
    additional_messages: tuple[ConversationItem, ...] = ()


type BeforeRunHook = Callable[
    [BeforeRunContext],
    Awaitable[BeforeRunResult | None],
]


class _BeforeRunExecution:
    def __init__(self, hook: BeforeRunHook) -> None:
        self._hook = hook

    async def apply(self, context: BeforeRunContext) -> BeforeRunContext:
        result = await self._hook(context.model_copy(deep=True))
        result = _validate_before_run_result(context, result)
        return _apply_before_run_result(context, result)


def _validate_before_run_result(
    context: BeforeRunContext,
    result: object,
) -> BeforeRunResult | None:
    if result is None:
        return None
    if not isinstance(result, BeforeRunResult):
        raise TypeError("before_run hooks must return BeforeRunResult or None.")
    validated = BeforeRunResult.model_validate_json(result.model_dump_json())
    _validate_tool_exchanges((*context.messages, *validated.additional_messages))
    return validated


def _apply_before_run_result(
    context: BeforeRunContext,
    result: BeforeRunResult | None,
) -> BeforeRunContext:
    if result is None:
        return context
    return BeforeRunContext(
        session_id=context.session_id,
        run_id=context.run_id,
        system_prompt=_result_system_prompt(context, result),
        messages=(*context.messages, *result.additional_messages),
    )


def _validate_tool_exchanges(messages: Sequence[ConversationItem]) -> None:
    calls: dict[str, str] = {}
    answered: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantTurn):
            _collect_tool_calls(message, calls)
        elif isinstance(message, ToolResultTurn):
            _validate_tool_result(message, calls, answered)
    unanswered = calls.keys() - answered
    if unanswered:
        call_ids = ", ".join(sorted(unanswered))
        raise ValueError(f"before_run tool calls require results: {call_ids}")


def _collect_tool_calls(turn: AssistantTurn, calls: dict[str, str]) -> None:
    for block in turn.blocks:
        if not isinstance(block, ToolCallBlock):
            continue
        if block.call_id in calls:
            raise ValueError(f"Duplicate before_run tool call: {block.call_id}")
        calls[block.call_id] = block.name


def _validate_tool_result(
    result: ToolResultTurn,
    calls: dict[str, str],
    answered: set[str],
) -> None:
    tool_name = calls.get(result.call_id)
    if tool_name is None:
        raise ValueError(f"Unknown before_run tool call: {result.call_id}")
    if result.call_id in answered:
        raise ValueError(f"Duplicate before_run tool result: {result.call_id}")
    if result.tool_name != tool_name:
        raise ValueError(
            f"Tool result {result.call_id!r} names {result.tool_name!r}, "
            f"expected {tool_name!r}."
        )
    answered.add(result.call_id)


def _result_system_prompt(
    context: BeforeRunContext,
    result: BeforeRunResult,
) -> str:
    if result.system_prompt is None:
        return context.system_prompt
    return result.system_prompt

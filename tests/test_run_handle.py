"""Tests for the persistence-free live RunHandle boundary."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tile import Completed, ExecutionFailure, Failed, RunHandle, RunRecord
from tile.events import RunEndEvent
from tile.runtime.execution import _ExecutionDependencies
from tile.runtime.handle import (
    _RunCompletion,
    _RunFinalization,
    _terminal_outcome,
)
from tile.tool_executor import ToolExecutor
from tests.support.agent_streams import ProviderStreamMock, final_text_stream


def test_run_handle_delegates_completion_without_a_store() -> None:
    """Pass immutable completion data to application-owned finalization."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    completions: list[_RunCompletion] = []

    def finish(completion: _RunCompletion) -> _RunFinalization:
        """Simulate the runtime's persistence callback."""

        completions.append(completion)
        record = completion.record.finish(outcome=completion.outcome)
        return _RunFinalization(record=record, outcome=completion.outcome)

    async def _run() -> RunHandle:
        handle = RunHandle(
            record=_running_record(),
            committed_history=(),
            result=None,
            execution=_execution(provider),
            on_finished=finish,
        )
        assert await handle.wait() == "completed"
        return handle

    handle = asyncio.run(_run())

    assert handle.record.status == "completed"
    assert handle.outcome == Completed(value="done")
    assert [item.role for item in completions[0].history_delta] == [
        "user",
        "assistant",
    ]


def test_run_handle_closes_when_finalization_callback_raises() -> None:
    """Preserve event-log closure when application orchestration violates its contract."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])

    def fail_finalization(completion: _RunCompletion) -> _RunFinalization:
        """Raise instead of returning the required finalization value."""

        _ = completion
        raise RuntimeError("finalization unavailable")

    async def _run() -> tuple[RunHandle, list[RunEndEvent]]:
        handle = RunHandle(
            record=_running_record(),
            committed_history=(),
            result=None,
            execution=_execution(provider),
            on_finished=fail_finalization,
        )
        with pytest.raises(RuntimeError, match="finalization unavailable"):
            await handle.wait()
        terminal = [
            event async for event in handle.events() if isinstance(event, RunEndEvent)
        ]
        return handle, terminal

    handle, terminal = asyncio.run(_run())

    assert handle.status == "running"
    assert len(terminal) == 1
    assert isinstance(terminal[0].outcome, Failed)
    assert isinstance(terminal[0].outcome.cause, ExecutionFailure)


def test_cancelled_task_requires_an_explicit_abort_reason() -> None:
    """Reject task cancellation that bypasses the RunHandle lifecycle."""

    async def _run() -> None:
        task: asyncio.Task[Completed] = asyncio.create_task(_pending_outcome())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(RuntimeError, match="missing an abort reason"):
            _terminal_outcome(task, abort_reason=None)

    asyncio.run(_run())


async def _pending_outcome() -> Completed:
    """Remain pending until the test cancels the task."""

    await asyncio.Event().wait()
    return Completed(value="unreachable")


def _running_record() -> RunRecord:
    """Build one accepted persistent run snapshot."""

    return RunRecord(
        run_id="run-1",
        session_id="session-1",
        prompt="hello",
        status="running",
        started_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        model="gpt-5.4",
        provider="test",
    )


def _execution(provider: ProviderStreamMock) -> _ExecutionDependencies:
    """Build pure execution dependencies without a Store."""

    return _ExecutionDependencies(
        stream_fn=provider.fn,
        model="gpt-5.4",
        instructions="Test.",
        cwd=Path("."),
        auto_mode=False,
        tool_executor=ToolExecutor(()),
    )

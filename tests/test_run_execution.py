"""Tests for run execution ownership, its handle, and process-local results."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import get_args, override

from tests.support.agent_streams import ProviderStreamMock, final_text_stream
from tests.support.store import terminal_outcome
from tile import Aborted, Completed, RunOutcome, TerminalRun
from tile.events import AgentEvent, RunEndEvent, RunFaultEvent
from tile.extensions.hooks import RunHooks
from tile.extensions.run_observers import RunObservers
from tile.result import Faulted, RunResult
from tile.runtime.run_execution import RunExecution, _RunDependencies
from tile.runtime.run_handle import RunHandle
from tile.sessions import SessionRepository
from tile.store import SQLiteStore
from tile.tool_executor import ToolExecutor
from tile.types import ConversationItem


async def test_run_execution_persists_before_provider_and_returns_only_outcome(
    store: SQLiteStore,
) -> None:
    """Own durable admission, execution, finalization, and terminal events."""

    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "done")])

    execution = await RunExecution.start(
        session=session,
        prompt="hello",
        result_type=None,
        provider=transport,
        dependencies=_dependencies(),
        hooks=RunHooks(),
        observers=RunObservers(),
    )
    handle = RunHandle(execution)
    assert store.get_run(session_id=session.id, run_id=handle.id).status == "active"
    result, events = await _wait_and_collect(handle)

    assert result == Completed(value="done")
    assert isinstance(events[-1], RunEndEvent)
    assert handle.session_id == session.id
    assert (
        terminal_outcome(store.get_run(session_id=session.id, run_id=handle.id))
        == result
    )
    assert [item.role for item in session.get_history()] == ["user", "assistant"]
    assert tuple(vars(handle)) == ("_execution",)


async def test_run_execution_closes_after_an_unexpected_finalization_crash() -> None:
    """Release waiters and event subscribers for every finalization failure."""

    store = _CrashingFinishStore(in_memory=True)
    try:
        session = SessionRepository(store).create(session_id="session-1")
        transport = ProviderStreamMock([final_text_stream("response-1", "lost")])

        handle = RunHandle(
            await RunExecution.start(
                session=session,
                prompt="hello",
                result_type=None,
                provider=transport,
                dependencies=_dependencies(),
                hooks=RunHooks(),
                observers=RunObservers(),
            )
        )
        result, events = await asyncio.wait_for(_wait_and_collect(handle), timeout=1)

        assert isinstance(result, Faulted)
        assert isinstance(result.error, _FinalizationCrash)
        assert isinstance(events[-1], RunFaultEvent)
    finally:
        store.close()


async def test_cancelled_execution_defaults_a_missing_abort_reason(
    store: SQLiteStore,
) -> None:
    """Classify cancellation safely when no explicit reason was recorded."""

    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([])
    transport.mock.side_effect = asyncio.CancelledError()

    handle = RunHandle(
        await RunExecution.start(
            session=session,
            prompt="hello",
            result_type=None,
            provider=transport,
            dependencies=_dependencies(),
            hooks=RunHooks(),
            observers=RunObservers(),
        )
    )
    result, events = await _wait_and_collect(handle)

    assert result == Aborted(reason="cancelled")
    assert isinstance(events[-1], RunEndEvent)
    assert (
        terminal_outcome(store.get_run(session_id=session.id, run_id=handle.id))
        == result
    )


def test_run_result_adds_faulted_without_expanding_persisted_outcomes() -> None:
    """Keep durability faults outside the RunRecord outcome contract."""

    error = OSError("disk full")
    result = Faulted(error=error)

    assert result.error is error
    assert result.type == "faulted"
    assert Faulted not in get_args(RunOutcome)
    assert Faulted in get_args(RunResult)
    assert isinstance(result, RunResult)
    assert not isinstance(result, RunOutcome)


def test_run_fault_event_carries_serializable_error_details() -> None:
    """Expose a live terminal fault without serializing its exception object."""

    event = RunFaultEvent(exception_type="OSError", message="disk full")

    assert event.model_dump() == {
        "type": "run_fault",
        "exception_type": "OSError",
        "message": "disk full",
    }


async def _wait_and_collect(
    handle: RunHandle,
) -> tuple[RunOutcome | Faulted, list[AgentEvent]]:
    """Wait for one handle and collect its complete live event log."""

    result = await handle.wait()
    events = [event async for event in handle.events()]
    return result, events


def _dependencies() -> _RunDependencies:
    """Build execution dependencies around one configured fake provider."""

    return _RunDependencies(
        instructions="Test",
        cwd=Path(),
        tool_executor=ToolExecutor(()),
    )


class _FinalizationCrash(BaseException):
    """Unexpected failure outside the normal Store exception hierarchy."""


class _CrashingFinishStore(SQLiteStore):
    """Store that raises an unexpected base exception during finalization."""

    @override
    def finish_run(
        self,
        *,
        session_id: str,
        run_id: str,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
    ) -> TerminalRun:
        """Crash terminal persistence after successful admission."""

        _ = session_id, run_id, outcome, history_delta
        raise _FinalizationCrash("unexpected finalization crash")

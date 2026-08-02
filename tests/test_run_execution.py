"""Tests for run execution ownership and its caller-facing handle."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from tile import Aborted, Completed, RunOutcome, RunRecord, StorePersistenceError
from tile.events import AgentEvent, RunEndEvent, RunFaultEvent
from tile.result import Faulted
from tile.runtime.execution import _ExecutionDependencies
from tile.runtime.run_execution import RunExecution
from tile.runtime.run_handle import RunHandle
from tile.sessions import SessionRepository
from tile.store import SQLiteStore
from tile.tool_executor import ToolExecutor
from tile.types import ConversationItem
from tests.support.agent_streams import ProviderStreamMock, final_text_stream
from tests.support.store import FailingFinishStore


def test_run_execution_persists_before_provider_and_returns_only_outcome() -> None:
    """Own durable admission, execution, finalization, and terminal events."""

    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "done")])

    async def run() -> tuple[RunHandle, RunOutcome | Faulted, list[AgentEvent]]:
        """Start and finish execution inside its owning event loop."""

        execution = RunExecution.start(
            session=session,
            prompt="hello",
            result=None,
            dependencies=_dependencies(transport),
        )
        handle = RunHandle(execution)
        assert (
            store.get_run(session_id=session.id, run_id=handle.id).status == "running"
        )
        result, events = await _wait_and_collect(handle)
        return handle, result, events

    handle, result, events = asyncio.run(run())

    assert result == Completed(value="done")
    assert isinstance(events[-1], RunEndEvent)
    assert handle.session_id == session.id
    assert store.get_run(session_id=session.id, run_id=handle.id).outcome == result
    assert [item.role for item in session.get_history()] == ["user", "assistant"]
    assert tuple(vars(handle)) == ("_execution",)
    store.close()


def test_run_execution_returns_faulted_when_finalization_is_not_durable() -> None:
    """Replace a candidate outcome with Faulted after Store failure."""

    store = FailingFinishStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "lost")])

    async def run() -> tuple[RunHandle, RunOutcome | Faulted, list[AgentEvent]]:
        """Start and fault execution inside its owning event loop."""

        handle = RunHandle(
            RunExecution.start(
                session=session,
                prompt="hello",
                result=None,
                dependencies=_dependencies(transport),
            )
        )
        result, events = await _wait_and_collect(handle)
        return handle, result, events

    handle, result, events = asyncio.run(run())

    assert isinstance(result, Faulted)
    assert isinstance(result.error, StorePersistenceError)
    assert isinstance(events[-1], RunFaultEvent)
    assert store.get_run(session_id=session.id, run_id=handle.id).status == "running"
    assert session.get_history() == ()
    store.close()


def test_run_execution_closes_after_an_unexpected_finalization_crash() -> None:
    """Release waiters and event subscribers for every finalization failure."""

    store = _CrashingFinishStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "lost")])

    async def run() -> tuple[RunOutcome | Faulted, list[AgentEvent]]:
        """Wait for the crash to become a terminal local fault."""

        handle = RunHandle(
            RunExecution.start(
                session=session,
                prompt="hello",
                result=None,
                dependencies=_dependencies(transport),
            )
        )
        return await asyncio.wait_for(_wait_and_collect(handle), timeout=1)

    result, events = asyncio.run(run())

    assert isinstance(result, Faulted)
    assert isinstance(result.error, _FinalizationCrash)
    assert isinstance(events[-1], RunFaultEvent)
    store.close()


def test_cancelled_execution_defaults_a_missing_abort_reason() -> None:
    """Classify cancellation safely when no explicit reason was recorded."""

    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([])
    transport.mock.side_effect = asyncio.CancelledError()

    async def run() -> tuple[RunHandle, RunOutcome | Faulted, list[AgentEvent]]:
        """Let provider cancellation bypass the explicit handle abort path."""

        handle = RunHandle(
            RunExecution.start(
                session=session,
                prompt="hello",
                result=None,
                dependencies=_dependencies(transport),
            )
        )
        result, events = await _wait_and_collect(handle)
        return handle, result, events

    handle, result, events = asyncio.run(run())

    assert result == Aborted(reason="cancelled")
    assert isinstance(events[-1], RunEndEvent)
    assert store.get_run(session_id=session.id, run_id=handle.id).outcome == result
    store.close()


async def _wait_and_collect(
    handle: RunHandle,
) -> tuple[RunOutcome | Faulted, list[AgentEvent]]:
    """Wait for one handle and collect its complete live event log."""

    result = await handle.wait()
    events = [event async for event in handle.events()]
    return result, events


def _dependencies(transport: ProviderStreamMock) -> _ExecutionDependencies:
    """Build execution dependencies around one configured fake provider."""

    return _ExecutionDependencies(
        provider=transport,
        instructions="Test",
        cwd=Path("."),
        auto_mode=False,
        tool_executor=ToolExecutor(()),
    )


class _FinalizationCrash(BaseException):
    """Unexpected failure outside the normal Store exception hierarchy."""


class _CrashingFinishStore(SQLiteStore):
    """Store that raises an unexpected base exception during finalization."""

    def finish_run(
        self,
        *,
        session_id: str,
        run_id: str,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord:
        """Crash terminal persistence after successful admission."""

        _ = session_id, run_id, outcome, history_delta
        raise _FinalizationCrash("unexpected finalization crash")

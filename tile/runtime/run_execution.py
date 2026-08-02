"""Live execution and durable finalization for one accepted run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

from pydantic import BaseModel

from tile.events import AgentEvent, RunEndEvent, RunFaultEvent, RunStartEvent
from tile.exceptions import TurnFailedError
from tile.result import (
    AbortReason,
    Aborted,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    Faulted,
    RunOutcome,
    RunResult,
)
from tile.runtime.execution import _ExecutionDependencies, execute_prompt
from tile.runtime.history import _RunHistory
from tile.sessions import Session
from tile.store.base import StaleRunError
from tile.store.models import HistoryItem, RunRecord


class RunExecution:
    """Own one run's live execution and durable lifecycle."""

    @classmethod
    def start(
        cls,
        *,
        session: Session,
        prompt: str,
        result: type[BaseModel] | None,
        dependencies: _ExecutionDependencies,
    ) -> RunExecution:
        """Durably accept one run, start execution, and return its owner."""

        record = RunRecord.start(
            id=str(uuid4()),
            session_id=session.id,
            prompt=prompt,
            model=dependencies.provider.model,
            provider=dependencies.provider.name,
        )
        started = session._start_run(record)
        execution = cls(
            session=session,
            record=started.run,
            committed_history=started.committed_history,
            result=result,
            dependencies=dependencies,
        )
        execution._begin()
        return execution

    def __init__(
        self,
        *,
        session: Session,
        record: RunRecord,
        committed_history: Sequence[HistoryItem],
        result: type[BaseModel] | None,
        dependencies: _ExecutionDependencies,
    ) -> None:
        """Prepare an already accepted run without performing side effects."""

        self._session = session
        self._record = record
        self._result_type = result
        self._dependencies = dependencies
        self._events: list[AgentEvent] = []
        self._history = _RunHistory.start(committed_history, prompt=record.prompt)
        self._result: RunResult | None = None
        self._abort_reason: AbortReason | None = None
        self._changed = asyncio.Event()
        self._finalized = asyncio.Event()
        self._task: asyncio.Task[RunOutcome] | None = None
        self._emit(RunStartEvent())

    @property
    def id(self) -> str:
        """Return the stable persistent run id."""

        return self._record.id

    @property
    def session_id(self) -> str:
        """Return the session that owns this run."""

        return self._record.session_id

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield live events through the durable end or harness fault."""

        index = 0
        while True:
            self._changed.clear()
            while index < len(self._events):
                yield self._events[index]
                index += 1
            if self._finalized.is_set():
                return
            await self._changed.wait()

    async def wait(self) -> RunResult:
        """Wait for and return the run's outcome or harness fault."""

        await self._finalized.wait()
        if self._result is None:
            raise RuntimeError("Run finalized without producing a result.")
        return self._result

    def abort(self) -> None:
        """Request explicit cancellation of this run."""

        self._cancel(reason="cancelled")

    def _begin(self) -> None:
        """Start the task for an already durably accepted run."""

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(
            execute_prompt(
                self._emit,
                deps=self._dependencies,
                history=self._history.working,
                result=self._result_type,
            )
        )
        self._task.add_done_callback(self._finalize)

    def _cancel(self, *, reason: AbortReason) -> None:
        """Cancel unfinished execution with its durable abort reason."""

        task = self._require_task()
        if task.done():
            return
        self._abort_reason = reason
        task.cancel()

    def _emit(self, event: AgentEvent) -> None:
        """Publish one event and project its replayable history item."""

        self._history.observe(event)
        self._events.append(event)
        self._changed.set()

    def _finalize(self, task: asyncio.Task[RunOutcome]) -> None:
        """Persist a terminal outcome or expose a process-local fault."""

        try:
            result = self._finish(task)
        except BaseException as error:
            result = Faulted(error=error)
        self._result = result
        self._append_terminal_event(result)
        self._finalized.set()
        self._changed.set()

    def _finish(self, task: asyncio.Task[RunOutcome]) -> RunResult:
        """Derive, heal, and durably commit one terminal outcome."""

        outcome = _terminal_outcome(task, abort_reason=self._abort_reason)
        self._history.heal()
        try:
            record = self._session._finish_run(
                self.id,
                outcome=outcome,
                history_delta=self._history.conversation_items(),
            )
        except StaleRunError as error:
            return self._reconcile_stale(error)
        if record.outcome is None:
            raise RuntimeError("Store returned a running record after finalization.")
        return record.outcome

    def _reconcile_stale(self, stale_error: StaleRunError) -> RunResult:
        """Return the authoritative terminal outcome after a stale write."""

        record = self._session._get_run(self.id)
        if record.outcome is None:
            return Faulted(error=stale_error)
        return record.outcome

    def _append_terminal_event(self, result: RunResult) -> None:
        """Close the live event log with a durable end or harness fault."""

        if isinstance(result, Faulted):
            self._emit(
                RunFaultEvent(
                    exception_type=type(result.error).__name__,
                    message=str(result.error),
                )
            )
            return
        self._emit(RunEndEvent(outcome=result))

    def _require_task(self) -> asyncio.Task[RunOutcome]:
        """Return the started execution task."""

        if self._task is None:
            raise RuntimeError("Run execution has not started.")
        return self._task


def _terminal_outcome(
    task: asyncio.Task[RunOutcome],
    *,
    abort_reason: AbortReason | None,
) -> RunOutcome:
    """Derive a serializable outcome from task completion."""

    if task.cancelled():
        return _aborted_outcome(abort_reason)
    error = task.exception()
    if error is not None:
        return Failed(cause=_execution_failure(error))
    return task.result()


def _aborted_outcome(abort_reason: AbortReason | None) -> Aborted:
    """Require cancellation to originate from an explicit lifecycle action."""

    if abort_reason is None:
        raise RuntimeError("Cancelled run is missing an abort reason")
    return Aborted(reason=abort_reason)


def _execution_failure(error: BaseException) -> ExecutionFailure:
    """Serialize one runtime exception without retaining it in persistence."""

    return ExecutionFailure(
        origin=_execution_failure_origin(error),
        exception_type=type(error).__name__,
        message=str(error),
    )


def _execution_failure_origin(error: BaseException) -> ExecutionFailureOrigin:
    """Classify an execution failure at the narrowest known boundary."""

    if isinstance(error, TurnFailedError):
        return "turn"
    return "execution"

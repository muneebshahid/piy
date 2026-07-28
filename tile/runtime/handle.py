"""Persistence-free live execution handle for one accepted run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from tile.events import (
    AgentEvent,
    RunEndEvent,
    RunStartEvent,
)
from tile.result import (
    AbortReason,
    Aborted,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    RunOutcome,
)
from tile.runtime.execution import (
    TurnFailedError,
    _ExecutionDependencies,
    execute_prompt,
)
from tile.runtime.history import _RunHistory
from tile.runtime.report import RunReport
from tile.store.models import HistoryItem, RunRecord
from tile.types.conversation import ConversationItem


@dataclass(frozen=True)
class _RunCompletion:
    """Execution result prepared for application-owned finalization."""

    record: RunRecord
    history_delta: tuple[ConversationItem, ...]


_OnFinished = Callable[[_RunCompletion], RunRecord]


class RunHandle:
    """Own live execution, provisional history, and event delivery."""

    def __init__(
        self,
        *,
        record: RunRecord,
        committed_history: Sequence[HistoryItem],
        result: type[BaseModel] | None,
        execution: _ExecutionDependencies,
        on_finished: _OnFinished,
    ) -> None:
        """Start execution and delegate durable finalization to the callback."""

        self._initial_record = record
        self._on_finished = on_finished
        self._events: list[AgentEvent] = [RunStartEvent()]
        self._history = _RunHistory.start(
            committed_history,
            prompt=record.prompt,
        )
        self._report: RunReport | None = None
        self._abort_reason: AbortReason | None = None
        self._changed = asyncio.Event()
        self._finalized = asyncio.Event()
        self._task = asyncio.create_task(
            execute_prompt(
                self._publish,
                deps=execution,
                history=self._history.working,
                result=result,
            )
        )
        self._task.add_done_callback(self._finalize)

    @property
    def id(self) -> str:
        """Return the stable run id."""

        return self._initial_record.run_id

    @property
    def session_id(self) -> str:
        """Return the session this run belongs to."""

        return self._initial_record.session_id

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield run events from the start through exactly one terminal event."""

        index = 0
        while True:
            self._changed.clear()
            while index < len(self._events):
                yield self._events[index]
                index += 1
            if self._finalized.is_set():
                return
            await self._changed.wait()

    async def wait(self) -> RunReport:
        """Wait for and return the complete terminal report without lifecycle raises."""

        await self._finalized.wait()
        if self._report is None:
            raise RuntimeError("Run finalized without producing a report.")
        return self._report

    def abort(self) -> None:
        """Request explicit cancellation of this run."""

        self._cancel(reason="cancelled")

    def _replace(self) -> None:
        """Cancel local execution after the Store atomically replaced this run."""

        self._cancel(reason="replaced")

    def _cancel(self, *, reason: AbortReason) -> None:
        """Cancel unfinished execution with its durable abort reason."""

        if self._task.done():
            return
        self._abort_reason = reason
        self._task.cancel()

    def _publish(self, event: AgentEvent) -> None:
        """Publish an event and project its replayable item into local history."""

        self._events.append(event)
        self._history.observe(event)
        self._changed.set()

    def _finalize(self, task: asyncio.Task[RunOutcome]) -> None:
        """Delegate persistence and always release terminal-state waiters."""

        try:
            outcome, execution_error = _terminal_outcome(
                task,
                abort_reason=self._abort_reason,
            )
            self._history.heal()
            history_delta = self._history.conversation_items()
            report = self._persist(
                record=self._initial_record.finish(outcome=outcome),
                history_delta=history_delta,
                execution_error=execution_error,
            )
            self._apply_report(report)
        finally:
            self._finalized.set()
            self._changed.set()

    def _persist(
        self,
        *,
        record: RunRecord,
        history_delta: tuple[ConversationItem, ...],
        execution_error: BaseException | None,
    ) -> RunReport:
        """Persist through the application callback or retain a local result."""

        completion = _RunCompletion(
            record=record,
            history_delta=history_delta,
        )
        try:
            return RunReport(
                record=self._on_finished(completion),
                history_delta=history_delta,
                execution_error=execution_error,
            )
        except Exception as error:
            return RunReport(
                record=record,
                history_delta=history_delta,
                execution_error=execution_error,
                finalization_error=error,
            )

    def _apply_report(self, report: RunReport) -> None:
        """Store the terminal report and close the event log exactly once."""

        self._report = report
        self._events.append(RunEndEvent(outcome=report.outcome))
        self._changed.set()


def _terminal_outcome(
    task: asyncio.Task[RunOutcome],
    *,
    abort_reason: AbortReason | None,
) -> tuple[RunOutcome, BaseException | None]:
    """Derive a serializable terminal outcome from task completion."""

    if task.cancelled():
        return _aborted_outcome(abort_reason), None
    error = task.exception()
    if error is not None:
        failure = _execution_failure(error, _execution_failure_origin(error))
        return Failed(cause=failure), error
    return task.result(), None


def _aborted_outcome(abort_reason: AbortReason | None) -> Aborted:
    """Require cancellation to originate from an explicit lifecycle action."""

    if abort_reason is None:
        raise RuntimeError("Cancelled run is missing an abort reason")
    return Aborted(reason=abort_reason)


def _execution_failure_origin(error: BaseException) -> ExecutionFailureOrigin:
    """Classify an execution failure at the narrowest known boundary."""

    if isinstance(error, TurnFailedError):
        return "turn"
    return "execution"


def _execution_failure(
    error: BaseException,
    origin: ExecutionFailureOrigin,
) -> ExecutionFailure:
    """Serialize one runtime exception without retaining it in persistence."""

    return ExecutionFailure(
        origin=origin,
        exception_type=type(error).__name__,
        message=str(error),
    )

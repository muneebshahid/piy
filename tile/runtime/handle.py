"""Persistence-free live execution handle for one accepted run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from tile.events import (
    AgentEvent,
    MessageEndEvent,
    RunEndEvent,
    RunStartEvent,
)
from tile.result import (
    AbortReason,
    Aborted,
    AgentFailure,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    FailureCause,
    PersistenceFailure,
    RunOutcome,
)
from tile.runtime.execution import (
    TurnFailedError,
    _assistant_text,
    _ExecutionDependencies,
    execute_prompt,
)
from tile.runtime.history import _RunHistory
from tile.store.models import HistoryItem, RunRecord, RunStatus
from tile.types.conversation import ConversationItem


@dataclass(frozen=True)
class _RunCompletion:
    """Execution result prepared for application-owned finalization."""

    record: RunRecord
    outcome: RunOutcome
    history_delta: tuple[ConversationItem, ...]

    @property
    def run_id(self) -> str:
        """Return the persistent run identifier."""

        return self.record.run_id


@dataclass(frozen=True)
class _RunFinalization:
    """Application resolution applied to the live handle and event log."""

    record: RunRecord
    outcome: RunOutcome
    wait_error: BaseException | None = None


_OnFinished = Callable[[_RunCompletion], _RunFinalization]


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

        self._record = record
        self._on_finished = on_finished
        self._events: list[AgentEvent] = [RunStartEvent()]
        self._history = _RunHistory.start(
            committed_history,
            prompt=record.prompt,
        )
        self._exception: BaseException | None = None
        self._outcome: RunOutcome | None = None
        self._wait_error: BaseException | None = None
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

        return self._record.run_id

    @property
    def session_id(self) -> str:
        """Return the session this run belongs to."""

        return self._record.session_id

    @property
    def status(self) -> RunStatus:
        """Return the latest persistent lifecycle snapshot."""

        return self.record.status

    @property
    def record(self) -> RunRecord:
        """Return the persistent record supplied at start or finalization."""

        return self._record

    @property
    def error_message(self) -> str | None:
        """Return a concise terminal failure message when present."""

        failure = self.failure
        if isinstance(failure, AgentFailure):
            return failure.reason
        if isinstance(failure, ExecutionFailure | PersistenceFailure):
            return failure.message
        return None

    @property
    def failure(self) -> FailureCause | None:
        """Return the structured terminal failure cause when present."""

        if isinstance(self._outcome, Failed):
            return self._outcome.cause
        return None

    @property
    def exception(self) -> BaseException | None:
        """Return the original in-process execution exception."""

        return self._exception

    @property
    def output_text(self) -> str | None:
        """Return text from the latest completed assistant message."""

        for event in reversed(self._events):
            if isinstance(event, MessageEndEvent):
                return _assistant_text(event.assistant_turn)
        return None

    @property
    def outcome(self) -> RunOutcome | None:
        """Return the live log's terminal outcome after finalization."""

        return self._outcome

    @property
    def conversation_items(self) -> tuple[ConversationItem, ...]:
        """Return defensive snapshots of this run's provisional history delta."""

        return self._history.conversation_items()

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

    async def wait(self) -> RunStatus:
        """Wait for finalization, raising when the atomic commit failed."""

        await self._finalized.wait()
        if self._wait_error is not None:
            raise self._wait_error
        return self.status

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
        """Delegate terminal persistence, then close the live event log."""

        try:
            outcome, self._exception = _terminal_outcome(
                task,
                abort_reason=self._abort_reason,
            )
            self._history.heal()
            finalization = self._on_finished(self._completion(outcome))
        except BaseException as error:
            self._exception = self._exception or error
            finalization = _failed_finalization(self._record, error)
        finally:
            self._apply_finalization(finalization)
            self._finalized.set()
            self._changed.set()

    def _completion(self, outcome: RunOutcome) -> _RunCompletion:
        """Build the immutable value passed to application orchestration."""

        return _RunCompletion(
            record=self._record,
            outcome=outcome,
            history_delta=self._history.conversation_items(),
        )

    def _apply_finalization(self, finalization: _RunFinalization) -> None:
        """Apply the runtime's durable resolution and close exactly once."""

        self._record = finalization.record
        self._outcome = finalization.outcome
        self._wait_error = finalization.wait_error
        self._events.append(RunEndEvent(outcome=finalization.outcome))
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


def _failed_finalization(
    record: RunRecord,
    error: BaseException,
) -> _RunFinalization:
    """Close safely when local finalization violates its no-raise contract."""

    failure = ExecutionFailure(
        origin="execution",
        exception_type=type(error).__name__,
        message=f"Run finalization failed: {error}",
    )
    return _RunFinalization(
        record=record,
        outcome=Failed(cause=failure),
        wait_error=error,
    )


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

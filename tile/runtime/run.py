"""Live run execution over an atomic persistent lifecycle."""

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
from tile.store import (
    HistoryItem,
    RunPersistenceError,
    RunRecord,
    RunStatus,
    StaleRunError,
    Store,
)
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderSource


@dataclass(frozen=True)
class _RunSpec:
    """Execution-only options for one already-persisted run."""

    result: type[BaseModel] | None


@dataclass(frozen=True)
class _RunDependencies:
    """Dependencies used by a live handle and its prompt program."""

    execution: _ExecutionDependencies
    store: Store


class RunHandle:
    """Own one live execution while the Store owns its durable lifecycle."""

    def __init__(
        self,
        *,
        record: RunRecord,
        committed_history: Sequence[HistoryItem],
        spec: _RunSpec,
        deps: _RunDependencies,
        on_finished: Callable[[RunHandle], None],
    ) -> None:
        """Start execution from committed history and the persisted prompt."""

        self._record = record
        self._spec = spec
        self._deps = deps
        self._on_finished = on_finished
        self._events: list[AgentEvent] = [RunStartEvent()]
        self._history = _RunHistory.start(
            committed_history,
            prompt=record.prompt,
        )
        self._exception: BaseException | None = None
        self._outcome: RunOutcome | None = None
        self._finalization_error: RunPersistenceError | None = None
        self._abort_reason: str = "cancelled"
        self._changed = asyncio.Event()
        self._finalized = asyncio.Event()
        self._task = asyncio.create_task(
            execute_prompt(
                self._publish,
                deps=deps.execution,
                history=self._history.working,
                result=spec.result,
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
        """Return the authoritative stored status."""

        return self.record.status

    @property
    def record(self) -> RunRecord:
        """Return the current authoritative persistent run record."""

        return self._deps.store.get_run(self.id)

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

        return self._history.snapshot()

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
        if self._finalization_error is not None:
            raise self._finalization_error
        return self.status

    def abort(self) -> None:
        """Request explicit cancellation of this run."""

        self._cancel(reason="cancelled")

    def _replace(self) -> None:
        """Cancel local execution after the Store atomically replaced this run."""

        self._cancel(reason="replaced")

    def _cancel(self, *, reason: str) -> None:
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
        """Commit the terminal record and history, then close the live log."""

        outcome, exception = _terminal_outcome(
            task,
            abort_reason=self._abort_reason,
        )
        self._exception = exception
        try:
            self._commit(outcome)
        except StaleRunError:
            try:
                self._close_from_stored_record()
            except BaseException as error:
                self._close_with_persistence_failure(error)
        except BaseException as error:
            self._close_with_persistence_failure(error)
        finally:
            try:
                self._on_finished(self)
            finally:
                self._finalized.set()
                self._changed.set()

    def _commit(self, outcome: RunOutcome) -> None:
        """Atomically persist the outcome and healed replayable history."""

        source = _latest_provider_source(self._events)
        self._history.heal()
        self._record = self._deps.store.finish_run(
            run_id=self.id,
            outcome=outcome,
            history_delta=self._history.delta,
            provider=source.provider if source is not None else None,
            model=source.model if source is not None else None,
        )
        self._close_log(outcome)

    def _close_from_stored_record(self) -> None:
        """Close with the terminal outcome already committed by another owner."""

        self._record = self._deps.store.get_run(self.id)
        if self._record.outcome is None:
            raise StaleRunError(f"Run is stale without a terminal outcome: {self.id}")
        self._close_log(self._record.outcome)

    def _close_with_persistence_failure(self, error: BaseException) -> None:
        """Close visibly and retain a raising error for waiters."""

        failure = PersistenceFailure(
            operation="finish_run",
            exception_type=type(error).__name__,
            message=str(error),
        )
        self._finalization_error = RunPersistenceError(self.id, error)
        self._close_log(Failed(cause=failure))

    def _close_log(self, outcome: RunOutcome) -> None:
        """Append the one terminal event and retain its live outcome."""

        self._outcome = outcome
        self._events.append(RunEndEvent(outcome=outcome))
        self._changed.set()


def _terminal_outcome(
    task: asyncio.Task[RunOutcome],
    *,
    abort_reason: str,
) -> tuple[RunOutcome, BaseException | None]:
    """Derive a serializable terminal outcome from task completion."""

    if task.cancelled():
        reason = "replaced" if abort_reason == "replaced" else "cancelled"
        return Aborted(reason=reason), None
    error = task.exception()
    if error is not None:
        failure = _execution_failure(error, _execution_failure_origin(error))
        return Failed(cause=failure), error
    return task.result(), None


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


def _latest_provider_source(
    events: Sequence[AgentEvent],
) -> ProviderSource | None:
    """Return the latest provider identity from a finalized message."""

    for event in reversed(events):
        if isinstance(event, MessageEndEvent):
            return event.assistant_turn.source
    return None

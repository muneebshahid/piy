"""Live execution and durable finalization for one accepted run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from uuid import uuid4

from pydantic import BaseModel

from tile.events import AgentEvent, RunEndEvent, RunFaultEvent, RunStartEvent
from tile.exceptions import TurnFailedError
from tile.extensions.hooks import BeforeRunContext, RunHooks
from tile.extensions.run_observers import RunEvent, RunObservers
from tile.prompt import build_system_prompt
from tile.providers.base import Provider
from tile.result import (
    Aborted,
    ExecutionFailure,
    Failed,
    Faulted,
    RunOutcome,
    RunResult,
)
from tile.runtime.execution import (
    _ExecutionDependencies,
    build_execution_dependencies,
    execute_prompt,
)
from tile.runtime.history import _RunHistory
from tile.sessions import Session
from tile.store.base import RunAlreadyEndedError, StoreError
from tile.store.models import ActiveRun, HistoryItem, TerminalRun
from tile.tool_executor import ToolExecutor
from tile.types.conversation import ConversationItem, UserMessage


@dataclass(frozen=True)
class _RunDependencies:
    """Harness configuration needed to admit and execute a run."""

    instructions: str
    cwd: Path
    tool_executor: ToolExecutor


class RunExecution:
    """Own one run's live execution and durable lifecycle."""

    @classmethod
    async def start(
        cls,
        *,
        session: Session,
        prompt: str,
        result_type: type[BaseModel] | None,
        provider: Provider,
        dependencies: _RunDependencies,
        hooks: RunHooks,
        observers: RunObservers,
    ) -> RunExecution:
        """Durably accept one run, start execution, and return its owner."""

        run_id = str(uuid4())
        execution_dependencies = build_execution_dependencies(
            provider=provider,
            system_prompt=build_system_prompt(
                dependencies.instructions,
                dependencies.cwd,
            ),
            tool_executor=dependencies.tool_executor,
            result_type=result_type,
        )
        context = await hooks.before_run(
            BeforeRunContext(
                session_id=session.id,
                run_id=run_id,
                system_prompt=execution_dependencies.system_prompt,
                messages=(UserMessage(content=prompt),),
            )
        )
        execution_dependencies = replace(
            execution_dependencies,
            system_prompt=context.system_prompt,
        )
        started = session._start_run(
            run_id=run_id,
            prompt=prompt,
            model=provider.model,
            provider=provider.name,
        )
        execution = cls(
            session=session,
            record=started.run,
            committed_history=started.committed_history,
            initial_messages=context.messages,
            dependencies=execution_dependencies,
            hooks=hooks,
            observers=observers,
        )
        execution._begin()
        return execution

    def __init__(
        self,
        *,
        session: Session,
        record: ActiveRun,
        committed_history: Sequence[HistoryItem],
        initial_messages: Sequence[ConversationItem],
        dependencies: _ExecutionDependencies,
        hooks: RunHooks,
        observers: RunObservers,
    ) -> None:
        """Prepare an already accepted run without performing side effects."""

        self._session: Final = session
        self._record: Final = record
        self._dependencies: Final = dependencies
        self._hooks: Final = hooks
        self._observers: Final = observers
        self._events: list[AgentEvent] = []
        self._history: Final = _RunHistory.start(
            committed_history,
            initial_messages=initial_messages,
        )
        self._result: RunResult | None = None
        self._changed: Final = asyncio.Event()
        self._finalized: Final = asyncio.Event()
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

        self._cancel()

    def _begin(self) -> None:
        """Start the task for an already durably accepted run."""

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(
            execute_prompt(
                self._emit,
                deps=self._dependencies,
                history=self._history.working,
            )
        )
        self._task.add_done_callback(self._finalize)

    def _cancel(self) -> None:
        """Cancel unfinished execution."""

        if self._task is None:
            raise RuntimeError("Run execution has not started.")
        if self._task.done():
            return
        self._task.cancel()

    def _emit(self, event: AgentEvent) -> None:
        """Publish one event and project its replayable history item."""

        self._history.observe(event)
        self._events.append(event)
        self._changed.set()
        self._observers.publish(
            RunEvent(session_id=self.session_id, run_id=self.id, event=event)
        )

    def _finalize(self, task: asyncio.Task[RunOutcome]) -> None:
        """Persist a terminal outcome or expose a process-local fault."""

        try:
            result = self._finish(task)
        except BaseException as error:  # noqa: BLE001
            result = Faulted(error=error)
        self._result = result
        self._append_terminal_event(result)
        self._finalized.set()

    def _finish(self, task: asyncio.Task[RunOutcome]) -> RunResult:
        """Derive, heal, and durably commit one terminal outcome."""

        outcome = self._terminal_outcome(task)
        self._history.heal()
        try:
            record = self._session._finish_run(
                self.id,
                outcome=outcome,
                history_delta=self._history.conversation_items(),
            )
        except RunAlreadyEndedError:
            return self._ended_outcome()
        return record.outcome

    def _terminal_outcome(self, task: asyncio.Task[RunOutcome]) -> RunOutcome:
        """Derive a serializable outcome from this execution's task state."""

        if task.cancelled():
            return Aborted()
        error = task.exception()
        if error is not None:
            return Failed(cause=_execution_failure(error))
        return task.result()

    def _ended_outcome(self) -> RunOutcome:
        """Return the authoritative outcome of an already-ended run."""

        record = self._session._get_run(self.id)
        if not isinstance(record, TerminalRun):
            raise StoreError(f"Run {self.id!r} ended without a terminal record.")
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


def _execution_failure(error: BaseException) -> ExecutionFailure:
    """Serialize one runtime exception without retaining it in persistence."""

    return ExecutionFailure(
        origin="turn" if isinstance(error, TurnFailedError) else "execution",
        exception_type=type(error).__name__,
        message=str(error),
    )

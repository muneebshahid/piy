"""Application service for persistent sessions and live prompt execution."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel

from tile.events import StreamFn
from tile.prompt import DEFAULT_INSTRUCTIONS
from tile.result import (
    COMPLETE_TOOL_NAME,
    FAIL_TOOL_NAME,
    Failed,
    PersistenceFailure,
    RunOutcome,
)
from tile.runtime.execution import _ExecutionDependencies
from tile.runtime.handle import RunHandle, _RunCompletion, _RunFinalization
from tile.runtime.session import Session
from tile.store.base import (
    RunPersistenceError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StaleRunError,
    Store,
)
from tile.store.models import RunRecord, SessionRecord
from tile.tool_executor import ToolExecutor
from tile.tools.support.paths import normalize_cwd
from tile.types.conversation import ConversationItem
from tile.types.tools import ToolDefinition, ToolFunction


class AgentRuntime:
    """Configure prompt execution over one authoritative Store."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn,
        model: str,
        cwd: Path | str,
        store: Store,
        tools: Sequence[ToolDefinition] = (),
        instructions: str = DEFAULT_INSTRUCTIONS,
        auto_mode: bool = True,
    ) -> None:
        """Create a runtime whose persistent mutations flow through one Store."""

        _reject_reserved_tool_names(tools)
        normalized_cwd = normalize_cwd(cwd)
        self._store = store
        self._execution = _ExecutionDependencies(
            stream_fn=stream_fn,
            model=model,
            instructions=instructions,
            cwd=normalized_cwd,
            auto_mode=auto_mode,
            tool_executor=ToolExecutor(_bind_cwd_tools(tools, normalized_cwd)),
        )
        self._active_runs: dict[str, RunHandle] = {}

    @property
    def sessions(self) -> tuple[Session, ...]:
        """Return handles for every persistent session."""

        return tuple(
            self._build_session(record) for record in self._store.list_sessions()
        )

    def session(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Create a session, or return the named existing session."""

        resolved_id = session_id if session_id is not None else str(uuid4())
        record = self._create_or_get_session(resolved_id, name=name)
        return self._build_session(record)

    def get_session(self, session_id: str) -> Session:
        """Return a handle for an existing persistent session."""

        return self._build_session(self._store.get_session(session_id))

    def history_for(self, session_id: str) -> tuple[ConversationItem, ...]:
        """Return defensive typed items from committed session history."""

        return tuple(
            envelope.item.model_copy(deep=True)
            for envelope in self._store.get_history(session_id)
        )

    def get_run(self, run_id: str) -> RunRecord:
        """Return an authoritative persistent run record."""

        return self._store.get_run(run_id)

    def runs_for(self, session_id: str) -> Sequence[RunRecord]:
        """Return persistent runs originating in one session."""

        return self._store.list_runs(session_id)

    def fork_session(
        self,
        *,
        source_session_id: str,
        target_session_id: str | None = None,
        name: str | None = None,
    ) -> Session:
        """Atomically fork all committed history into a new session."""

        record = self._store.fork_session(
            source_session_id=source_session_id,
            target_session_id=(
                target_session_id if target_session_id is not None else str(uuid4())
            ),
            name=name,
        )
        return self._build_session(record)

    def _submit_prompt(
        self,
        session_id: str,
        content: str,
        *,
        result: type[BaseModel] | None = None,
        replace_active: bool = False,
    ) -> RunHandle:
        """Persist a running record, then start its in-memory execution."""

        started = self._store.start_run(
            run_id=str(uuid4()),
            session_id=session_id,
            prompt=content,
            model=self._execution.model,
            provider=self._execution.stream_fn.provider,
            replace_active=replace_active,
        )
        self._cancel_replaced_local_run(started.replaced_run_id)
        handle = RunHandle(
            record=started.run,
            committed_history=started.committed_history,
            result=result,
            execution=self._execution,
            on_finished=self._finish_run,
        )
        self._active_runs[handle.id] = handle
        return handle

    def _create_or_get_session(
        self,
        session_id: str,
        *,
        name: str | None,
    ) -> SessionRecord:
        """Create one session while tolerating an existing caller-selected id."""

        try:
            return self._store.get_session(session_id)
        except SessionNotFoundError:
            try:
                return self._store.create_session(
                    session_id=session_id,
                    name=name,
                )
            except SessionAlreadyExistsError:
                return self._store.get_session(session_id)

    def _cancel_replaced_local_run(self, run_id: str | None) -> None:
        """Cancel a replaced handle when this runtime owns its task."""

        if run_id is None:
            return
        handle = self._active_runs.get(run_id)
        if handle is not None:
            handle._replace()

    def _finish_run(self, completion: _RunCompletion) -> _RunFinalization:
        """Persist one candidate completion and release its local handle."""

        try:
            return self._persist_completion(completion)
        finally:
            self._active_runs.pop(completion.run_id, None)

    def _persist_completion(self, completion: _RunCompletion) -> _RunFinalization:
        """Resolve successful, stale, and failed persistence paths."""

        try:
            record = self._store.finish_run(
                run_id=completion.run_id,
                outcome=completion.outcome,
                history_delta=completion.history_delta,
            )
        except StaleRunError:
            return self._resolve_stale_completion(completion)
        except BaseException as error:
            return _persistence_failure(completion, error)
        return _RunFinalization(record=record, outcome=completion.outcome)

    def _resolve_stale_completion(
        self,
        completion: _RunCompletion,
    ) -> _RunFinalization:
        """Use the terminal record committed by another lifecycle owner."""

        try:
            record = self._store.get_run(completion.run_id)
        except BaseException as error:
            return _persistence_failure(completion, error)
        outcome = cast("RunOutcome", record.outcome)
        return _RunFinalization(record=record, outcome=outcome)

    def _build_session(self, record: SessionRecord) -> Session:
        """Build the application facade for one persistent session."""

        return Session(_record=record, _runtime=self)


RESERVED_TOOL_NAMES = (COMPLETE_TOOL_NAME, FAIL_TOOL_NAME)


def _persistence_failure(
    completion: _RunCompletion,
    error: BaseException,
) -> _RunFinalization:
    """Return a visible failure while retaining the last durable snapshot."""

    failure = PersistenceFailure(
        operation="finish_run",
        exception_type=type(error).__name__,
        message=str(error),
    )
    return _RunFinalization(
        record=completion.record,
        outcome=Failed(cause=failure),
        wait_error=RunPersistenceError(completion.run_id, error),
    )


def _reject_reserved_tool_names(tools: Sequence[ToolDefinition]) -> None:
    """Reject caller tools whose names the output contract reserves."""

    for tool in tools:
        if tool.name.lower() in RESERVED_TOOL_NAMES:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved by the runtime for "
                "output contracts; rename the tool."
            )


def _bind_cwd_tools(
    tools: Sequence[ToolDefinition],
    cwd: Path,
) -> tuple[ToolDefinition, ...]:
    """Bind the runtime cwd into every tool that declares a cwd parameter."""

    return tuple(
        _bind_cwd(tool, cwd) if _expects_cwd(tool.fn) else tool for tool in tools
    )


def _bind_cwd(tool: ToolDefinition, cwd: Path) -> ToolDefinition:
    """Return a tool copy whose function receives the runtime cwd."""

    _reject_cwd_schema_property(tool)
    fn = cast(ToolFunction, partial(tool.fn, cwd=cwd))
    return tool.model_copy(update={"fn": fn})


def _expects_cwd(fn: ToolFunction) -> bool:
    """Return whether a tool function declares an explicit cwd parameter."""

    parameter = inspect.signature(fn).parameters.get("cwd")
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _reject_cwd_schema_property(tool: ToolDefinition) -> None:
    """Reject a schema that would expose the runtime-injected cwd."""

    properties = tool.input_model.model_json_schema().get("properties", {})
    if "cwd" in properties:
        raise ValueError(
            f"Tool '{tool.name}' declares cwd in its input schema; cwd is "
            "runtime-injected and must not be model-visible."
        )

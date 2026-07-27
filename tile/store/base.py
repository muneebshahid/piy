"""Atomic persistence boundary for sessions, runs, and committed history."""

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias

from tile.store.models import HistoryItem, RunRecord, SessionRecord, StartedRun
from tile.types.conversation import ConversationItem


class StoreError(RuntimeError):
    """Base error for persistent domain operations."""


class ActiveRunError(StoreError):
    """Raised when a session already owns a running run."""


class StaleRunError(StoreError):
    """Raised when a superseded or finalized run attempts to mutate state."""


class SessionAlreadyExistsError(StoreError, ValueError):
    """Raised when creating a session whose id already exists."""


class SessionNotFoundError(StoreError, KeyError):
    """Raised when an operation references an unknown session."""


class RunAlreadyExistsError(StoreError, ValueError):
    """Raised when creating a run whose id already exists."""


class RunNotFoundError(StoreError, KeyError):
    """Raised when an operation references an unknown run."""


StoreOperation: TypeAlias = Literal[
    "create_session",
    "get_session",
    "list_sessions",
    "start_run",
    "finish_run",
    "get_history",
    "get_run",
    "list_runs",
    "fork_session",
]


class StorePersistenceError(StoreError):
    """Raised when a Store backend cannot complete an operation."""

    def __init__(self, operation: StoreOperation, cause: BaseException) -> None:
        """Retain the failed Store operation and backend exception."""

        self.operation = operation
        self.cause = cause
        super().__init__(f"Store operation {operation!r} failed: {cause}")


class InvalidHistoryError(StoreError, ValueError):
    """Raised when conversation items cannot form replayable history."""


class Store(Protocol):
    """Own the atomic consistency boundary for persistent runtime state.

    Implementations must make each mutating method atomic. A backend that
    cannot provide equivalent all-or-nothing behavior is not a valid Store.
    Backend-specific failures must be raised as ``StorePersistenceError``;
    lifecycle conflicts and missing aggregates use the narrower domain errors.
    """

    def create_session(
        self,
        *,
        record: SessionRecord,
    ) -> SessionRecord:
        """Create and return one session, rejecting an existing id."""
        ...

    def get_session(self, session_id: str) -> SessionRecord:
        """Return one session or raise a session-not-found domain error."""
        ...

    def list_sessions(self) -> Sequence[SessionRecord]:
        """Return sessions in creation order."""
        ...

    def start_run(
        self,
        *,
        record: RunRecord,
        replace_active: bool = False,
    ) -> StartedRun:
        """Atomically start a run and snapshot its committed session history.

        The returned history must come from the same consistency boundary as
        the accepted run and optional predecessor replacement.
        """
        ...

    def finish_run(
        self,
        *,
        record: RunRecord,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord:
        """Atomically finalize a run and commit its replayable history delta.

        The caller owns construction of the terminal record and valid history
        delta. Implementations own stale-run fencing and all-or-nothing
        persistence.
        """
        ...

    def get_history(self, session_id: str) -> Sequence[HistoryItem]:
        """Return committed history items in session-local position order."""
        ...

    def get_run(self, run_id: str) -> RunRecord:
        """Return one persistent run or raise a run-not-found domain error."""
        ...

    def list_runs(self, session_id: str) -> Sequence[RunRecord]:
        """Return runs that originated in a session, in submission order."""
        ...

    def fork_session(
        self,
        *,
        source_session_id: str,
        target: SessionRecord,
    ) -> SessionRecord:
        """Atomically create a session with all committed source history."""
        ...

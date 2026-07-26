"""Atomic persistence boundary for sessions, runs, and committed history."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from tile.result import RunOutcome
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


class RunPersistenceError(StoreError):
    """Raised when a run cannot atomically persist its terminal state."""

    def __init__(self, run_id: str, cause: BaseException) -> None:
        """Retain the failing run and original persistence exception."""

        self.run_id = run_id
        self.cause = cause
        super().__init__(f"Could not finalize run {run_id}: {cause}")


class InvalidHistoryError(StoreError, ValueError):
    """Raised when conversation items cannot form replayable history."""


class Store(Protocol):
    """Own the atomic consistency boundary for persistent runtime state.

    Implementations must make each mutating method atomic. A backend that
    cannot provide equivalent all-or-nothing behavior is not a valid Store.
    """

    def create_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
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
        run_id: str,
        session_id: str,
        prompt: str,
        model: str,
        provider: str | None,
        replace_active: bool = False,
        started_at: datetime | None = None,
    ) -> StartedRun:
        """Atomically create a running run and optionally replace its predecessor."""
        ...

    def finish_run(
        self,
        *,
        run_id: str,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
        provider: str | None = None,
        model: str | None = None,
        ended_at: datetime | None = None,
    ) -> RunRecord:
        """Atomically finalize a still-running run and commit its history delta."""
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
        target_session_id: str,
        name: str | None = None,
        through_position: int | None = None,
    ) -> SessionRecord:
        """Atomically create a session with a copied flat history prefix."""
        ...

"""Store-bound session handles and the session collection repository."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import uuid4

from tile.result import RunOutcome
from tile.store.base import Store
from tile.store.models import RunRecord, SessionRecord, StartedRun
from tile.types.conversation import ConversationItem


@dataclass(frozen=True)
class Session:
    """Lightweight handle for one persistent session."""

    id: str
    _store: Store = field(repr=False, compare=False)

    def get_session_record(self) -> SessionRecord:
        """Return the current persistent metadata for this session."""

        return self._store.get_session(self.id)

    def get_history(self) -> Sequence[ConversationItem]:
        """Return the committed conversation history for this session."""

        return tuple(
            envelope.item.model_copy(deep=True)
            for envelope in self._store.get_history(self.id)
        )

    def get_runs(self) -> Sequence[RunRecord]:
        """Return the persistent runs owned by this session."""

        return self._store.list_runs(self.id)

    def _start_run(
        self,
        *,
        run_id: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> StartedRun:
        """Atomically accept one run and return its private bootstrap state."""

        return self._store.start_run(
            session_id=self.id,
            run_id=run_id,
            prompt=prompt,
            model=model,
            provider=provider,
        )

    def _finish_run(
        self,
        run_id: str,
        *,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord:
        """Atomically finish one run and append its replayable history."""

        return self._store.finish_run(
            session_id=self.id,
            run_id=run_id,
            outcome=outcome,
            history_delta=history_delta,
        )

    def _get_run(self, run_id: str) -> RunRecord:
        """Return one authoritative run for lifecycle reconciliation."""

        return self._store.get_run(self.id, run_id)


class SessionRepository:
    """Create and retrieve lightweight sessions through one Store."""

    def __init__(self, store: Store) -> None:
        """Bind the repository to its persistent Store."""

        self._store = store

    def create(
        self,
        *,
        session_id: str | None = None,
    ) -> Session:
        """Create and return a new persistent session."""

        resolved_id = session_id if session_id is not None else str(uuid4())
        persisted = self._store.create_session(session_id=resolved_id)
        return self._session(persisted.id)

    def get(self, session_id: str) -> Session:
        """Return a handle for an existing persistent session."""

        record = self._store.get_session(session_id)
        return self._session(record.id)

    def list(self) -> Sequence[Session]:
        """Return lightweight handles for every persistent session."""

        return tuple(self._session(record.id) for record in self._store.list_sessions())

    def abort_active_run(self, session_id: str) -> RunRecord | None:
        """Durably abort a running record without controlling its local task."""

        return self._store.abort_active_run(session_id)

    def fork(
        self,
        source_session_id: str,
        *,
        target_session_id: str | None = None,
    ) -> Session:
        """Fork committed history into a new persistent session."""

        resolved_target_id = (
            target_session_id if target_session_id is not None else str(uuid4())
        )
        persisted = self._store.fork_session(
            source_session_id=source_session_id,
            target_session_id=resolved_target_id,
        )
        return self._session(persisted.id)

    def _session(self, session_id: str) -> Session:
        """Build a lightweight handle bound to this repository's Store."""

        return Session(id=session_id, _store=self._store)

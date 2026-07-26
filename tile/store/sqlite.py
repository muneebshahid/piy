"""SQLite implementation of Tile's atomic persistence boundary."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from tile._sqlite import immediate_transaction, resolve_connection_target
from tile.result import Aborted, RunOutcome
from tile.store.base import (
    ActiveRunError,
    RunAlreadyExistsError,
    RunNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StaleRunError,
)
from tile.store.history import validate_replayable_history
from tile.store.models import (
    HistoryItem,
    RunRecord,
    SessionRecord,
    StartedRun,
)
from tile.store.schema import initialize_sqlite_schema
from tile.store.serialization import (
    HistoryRow,
    RunRow,
    SessionRow,
    dump_conversation_item,
    history_from_row,
    history_values,
    run_from_row,
    run_values,
    session_from_row,
    session_values,
    terminal_run_values,
)
from tile.types.conversation import ConversationItem


class SQLiteStore:
    """Persist sessions, runs, and committed history behind atomic operations."""

    def __init__(
        self,
        database_path: Path | str | None = None,
        *,
        in_memory: bool = False,
    ) -> None:
        """Open a SQLite database and initialize the unified schema."""

        target = resolve_connection_target(
            database_path=database_path,
            in_memory=in_memory,
        )
        self._connection = sqlite3.connect(target, isolation_level=None)
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            initialize_sqlite_schema(self._connection)
        except BaseException:
            self._connection.close()
            raise

    def create_session(
        self,
        *,
        session_id: str,
        name: str | None = None,
    ) -> SessionRecord:
        """Create and return one session, rejecting an existing id."""

        record = _new_session_record(session_id, name)
        try:
            with immediate_transaction(self._connection):
                self._insert_session(record)
        except sqlite3.IntegrityError as error:
            raise SessionAlreadyExistsError(
                f"Session already exists: {session_id}"
            ) from error
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        """Return one session or raise when it does not exist."""

        row = self._connection.execute(
            """
            SELECT id, name, created_at, updated_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")
        return session_from_row(cast("SessionRow", row))

    def list_sessions(self) -> tuple[SessionRecord, ...]:
        """Return sessions in creation order."""

        rows = self._connection.execute(
            """
            SELECT id, name, created_at, updated_at
            FROM sessions
            ORDER BY created_at, id
            """
        ).fetchall()
        session_rows = cast("Sequence[SessionRow]", rows)
        return tuple(session_from_row(row) for row in session_rows)

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
        """Atomically create a run and optionally replace its active predecessor."""

        record = _new_run_record(
            run_id=run_id,
            session_id=session_id,
            prompt=prompt,
            model=model,
            provider=provider,
            started_at=started_at,
        )
        with immediate_transaction(self._connection):
            self._require_session(session_id)
            active = self._running_run_for_session(session_id)
            replaced_run_id = self._replace_active_run(
                active,
                replace_active=replace_active,
                replaced_at=record.started_at,
            )
            committed_history = self._history_prefix(
                session_id,
                through_position=None,
            )
            self._insert_run(record)
        return StartedRun(
            run=record,
            committed_history=committed_history,
            replaced_run_id=replaced_run_id,
        )

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
        """Atomically finalize a still-running run and append its history delta."""

        with immediate_transaction(self._connection):
            running = self._get_run(run_id)
            if running.status != "running":
                raise StaleRunError(f"Run is no longer active: {run_id}")
            validate_replayable_history(history_delta)
            finished = running.finish(
                outcome=outcome,
                provider=provider,
                model=model,
                ended_at=ended_at,
            )
            self._update_running_run(finished)
            self._insert_history_delta(finished, history_delta)
            self._touch_session(finished)
        return finished

    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Return committed history in session-local order."""

        self._require_session(session_id)
        rows = self._connection.execute(
            """
            SELECT id, session_id, run_id, position, role, payload_json, created_at
            FROM history_items
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        history_rows = cast("Sequence[HistoryRow]", rows)
        return tuple(history_from_row(row) for row in history_rows)

    def get_run(self, run_id: str) -> RunRecord:
        """Return one persistent run or raise when it does not exist."""

        return self._get_run(run_id)

    def list_runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """Return runs originating in a session in submission order."""

        self._require_session(session_id)
        rows = self._connection.execute(
            _SELECT_RUNS_SQL + " WHERE session_id = ? ORDER BY started_at, id",
            (session_id,),
        ).fetchall()
        run_rows = cast("Sequence[RunRow]", rows)
        return tuple(run_from_row(row) for row in run_rows)

    def fork_session(
        self,
        *,
        source_session_id: str,
        target_session_id: str,
        name: str | None = None,
        through_position: int | None = None,
    ) -> SessionRecord:
        """Atomically create a session with a copied flat history prefix."""

        with immediate_transaction(self._connection):
            self._require_session(source_session_id)
            self._reject_existing_session(target_session_id)
            record = _new_session_record(target_session_id, name)
            self._insert_session(record)
            source_items = self._history_prefix(
                source_session_id,
                through_position=through_position,
            )
            if source_items:
                validate_replayable_history(tuple(item.item for item in source_items))
            self._copy_history_items(source_items, target_session_id)
        return record

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def _insert_session(self, record: SessionRecord) -> None:
        """Insert one validated session record."""

        self._connection.execute(
            """
            INSERT INTO sessions (id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            session_values(record),
        )

    def _require_session(self, session_id: str) -> None:
        """Raise when a session id is unknown."""

        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")

    def _reject_existing_session(self, session_id: str) -> None:
        """Raise when a session id is already present."""

        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is not None:
            raise SessionAlreadyExistsError(f"Session already exists: {session_id}")

    def _running_run_for_session(self, session_id: str) -> RunRecord | None:
        """Return the session's running run when present."""

        row = self._connection.execute(
            _SELECT_RUNS_SQL + " WHERE session_id = ? AND status = 'running'",
            (session_id,),
        ).fetchone()
        return None if row is None else run_from_row(cast("RunRow", row))

    def _replace_active_run(
        self,
        active: RunRecord | None,
        *,
        replace_active: bool,
        replaced_at: datetime,
    ) -> str | None:
        """Replace an active run or reject the conflicting start."""

        if active is None:
            return None
        if not replace_active:
            raise ActiveRunError(
                f"Session already has an active run: {active.session_id}"
            )
        replaced = active.finish(
            outcome=Aborted(reason="replaced"),
            ended_at=replaced_at,
        )
        self._update_running_run(replaced)
        return active.run_id

    def _insert_run(self, record: RunRecord) -> None:
        """Insert one running run with domain error translation."""

        try:
            self._connection.execute(_INSERT_RUN_SQL, run_values(record))
        except sqlite3.IntegrityError as error:
            if self._run_exists(record.run_id):
                raise RunAlreadyExistsError(
                    f"Run already exists: {record.run_id}"
                ) from error
            raise

    def _run_exists(self, run_id: str) -> bool:
        """Return whether a run id is already present."""

        row = self._connection.execute(
            "SELECT 1 FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        return row is not None

    def _get_run(self, run_id: str) -> RunRecord:
        """Return one run inside or outside a transaction."""

        row = self._connection.execute(
            _SELECT_RUNS_SQL + " WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Unknown run: {run_id}")
        return run_from_row(cast("RunRow", row))

    def _update_running_run(self, record: RunRecord) -> None:
        """Conditionally update a run only while it remains active."""

        cursor = self._connection.execute(
            _FINISH_RUN_SQL,
            terminal_run_values(record),
        )
        if cursor.rowcount == 0:
            raise StaleRunError(f"Run is no longer active: {record.run_id}")

    def _insert_history_delta(
        self,
        run: RunRecord,
        items: Sequence[ConversationItem],
    ) -> None:
        """Append one run's typed history delta in order."""

        if not items:
            return
        first_position = self._next_history_position(run.session_id)
        created_at = cast("datetime", run.ended_at)
        rows = [
            history_values(
                item=item,
                run_id=run.run_id,
                session_id=run.session_id,
                position=first_position + offset,
                created_at=created_at,
            )
            for offset, item in enumerate(items)
        ]
        self._connection.executemany(_INSERT_HISTORY_SQL, rows)

    def _touch_session(self, run: RunRecord) -> None:
        """Advance session metadata with the committed run timestamp."""

        if run.ended_at is None:
            raise ValueError("A committed run requires an end timestamp.")
        current = self.get_session(run.session_id)
        updated_at = max(current.updated_at, run.ended_at)
        self._connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (updated_at.isoformat(), run.session_id),
        )

    def _next_history_position(self, session_id: str) -> int:
        """Return the next session-local committed history position."""

        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1
            FROM history_items
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return cast("tuple[int]", row)[0]

    def _history_prefix(
        self,
        session_id: str,
        *,
        through_position: int | None,
    ) -> tuple[HistoryItem, ...]:
        """Return all history or the inclusive prefix ending at a position."""

        if through_position is not None and through_position < 0:
            raise ValueError("through_position cannot be negative.")
        sql = """
            SELECT id, session_id, run_id, position, role, payload_json, created_at
            FROM history_items
            WHERE session_id = ?
        """
        params: tuple[str] | tuple[str, int] = (session_id,)
        if through_position is not None:
            sql += " AND position <= ?"
            params = (session_id, through_position)
        rows = self._connection.execute(sql + " ORDER BY position", params).fetchall()
        history_rows = cast("Sequence[HistoryRow]", rows)
        return tuple(history_from_row(row) for row in history_rows)

    def _copy_history_items(
        self,
        items: Sequence[HistoryItem],
        target_session_id: str,
    ) -> None:
        """Duplicate history envelopes while retaining run provenance."""

        rows = [
            (
                str(uuid4()),
                target_session_id,
                item.run_id,
                item.position,
                item.item.role,
                dump_conversation_item(item.item),
                item.created_at.isoformat(),
            )
            for item in items
        ]
        self._connection.executemany(_INSERT_HISTORY_SQL, rows)


_RUN_COLUMNS = (
    "id",
    "session_id",
    "prompt",
    "status",
    "started_at",
    "ended_at",
    "model",
    "provider",
    "outcome_json",
)
_SELECT_RUNS_SQL = f"SELECT {', '.join(_RUN_COLUMNS)} FROM runs"
_INSERT_RUN_SQL = f"""
    INSERT INTO runs ({", ".join(_RUN_COLUMNS)})
    VALUES ({", ".join("?" for _ in _RUN_COLUMNS)})
"""
_FINISH_RUN_SQL = """
    UPDATE runs
    SET status = ?, ended_at = ?, model = ?, provider = ?, outcome_json = ?
    WHERE id = ? AND status = 'running'
"""
_INSERT_HISTORY_SQL = """
    INSERT INTO history_items (
        id,
        session_id,
        run_id,
        position,
        role,
        payload_json,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def _new_session_record(session_id: str, name: str | None) -> SessionRecord:
    """Create one session record with a stable creation timestamp."""

    now = datetime.now(UTC)
    return SessionRecord(
        session_id=session_id,
        name=name,
        created_at=now,
        updated_at=now,
    )


def _new_run_record(
    *,
    run_id: str,
    session_id: str,
    prompt: str,
    model: str,
    provider: str | None,
    started_at: datetime | None,
) -> RunRecord:
    """Create one validated running run record."""

    return RunRecord(
        run_id=run_id,
        session_id=session_id,
        prompt=prompt,
        status="running",
        started_at=started_at if started_at is not None else datetime.now(UTC),
        model=model,
        provider=provider,
    )

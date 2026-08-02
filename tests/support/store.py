"""Shared builders for SQLite Store behavior tests."""

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from tile import (
    Aborted,
    Completed,
    Failed,
    RunOutcome,
    RunRecord,
    StorePersistenceError,
)
from tile.store import SessionRecord, SQLiteStore, StartedRun
from tile.types import ConversationItem

STARTED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


class FailingStartStore(SQLiteStore):
    """SQLite Store with a recoverable deterministic admission failure."""

    fail_starts: bool = True

    def start_run(
        self,
        *,
        session_id: str,
        run_id: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> StartedRun:
        """Fail or delegate one atomic start operation."""

        if self.fail_starts:
            raise StorePersistenceError("start_run", OSError("disk full"))
        return super().start_run(
            session_id=session_id,
            run_id=run_id,
            prompt=prompt,
            model=model,
            provider=provider,
        )


class FailingFinishStore(SQLiteStore):
    """SQLite Store with a recoverable deterministic finalization failure."""

    fail_finishes: bool = True

    def finish_run(
        self,
        *,
        session_id: str,
        run_id: str,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord:
        """Fail or delegate one atomic finish operation."""

        if self.fail_finishes:
            raise StorePersistenceError("finish_run", OSError("disk full"))
        return super().finish_run(
            session_id=session_id,
            run_id=run_id,
            outcome=outcome,
            history_delta=history_delta,
        )


def start_run(
    store: SQLiteStore,
    *,
    session_id: str = "session-1",
    run_id: str = "run-1",
    prompt: str = "hello",
    model: str = "gpt-5.4",
    provider: str = "test",
) -> StartedRun:
    """Start one run from caller-owned execution inputs."""

    return store.start_run(
        session_id=session_id,
        run_id=run_id,
        prompt=prompt,
        model=model,
        provider=provider,
    )


def persist_outcome(
    store: SQLiteStore,
    *,
    outcome: Completed | Failed | Aborted,
    history_delta: Sequence[ConversationItem],
    session_id: str = "session-1",
    run_id: str = "run-1",
) -> RunRecord:
    """Finish and persist one Store-owned running record."""

    return store.finish_run(
        session_id=session_id,
        run_id=run_id,
        outcome=outcome,
        history_delta=history_delta,
    )


def create_session(
    store: SQLiteStore,
    *,
    session_id: str,
) -> SessionRecord:
    """Create one session from its caller-owned identity."""

    return store.create_session(session_id=session_id)


def seed_database(
    database_path: Path,
    *,
    session_id: str = "session-1",
    running_run: bool = False,
    run_id: str = "run-1",
) -> None:
    """Seed a file-backed database, then close the seeding store."""

    seed = SQLiteStore(database_path)
    try:
        create_session(seed, session_id=session_id)
        if running_run:
            start_run(seed, session_id=session_id, run_id=run_id)
    finally:
        seed.close()


def corrupt_column(
    database_path: Path,
    *,
    table: str,
    column: str,
    value: str = "not-json",
) -> None:
    """Overwrite one persisted column with raw corrupt data."""

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(f"UPDATE {table} SET {column} = ?", (value,))
        connection.commit()
    finally:
        connection.close()

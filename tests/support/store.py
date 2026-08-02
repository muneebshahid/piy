"""Shared builders for SQLite Store behavior tests."""

from collections.abc import Sequence
from datetime import UTC, datetime

from tile import Aborted, Completed, Failed, RunRecord
from tile.store import SQLiteStore, SessionRecord, StartedRun
from tile.types import ConversationItem

STARTED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def start_run(
    store: SQLiteStore,
    *,
    session_id: str = "session-1",
) -> StartedRun:
    """Start one run through the record-based Store contract."""

    return store.start_run(
        record=run_record(run_id="run-1", session_id=session_id),
    )


def persist_outcome(
    store: SQLiteStore,
    *,
    outcome: Completed | Failed | Aborted,
    history_delta: Sequence[ConversationItem],
    run_id: str = "run-1",
) -> RunRecord:
    """Finish and persist one Store-owned running record."""

    return store.finish_run(
        run_id=run_id,
        outcome=outcome,
        history_delta=history_delta,
    )


def running_record() -> RunRecord:
    """Build one running record without Store access."""

    return run_record(run_id="run-1", session_id="session-1")


def run_record(
    *,
    run_id: str,
    session_id: str,
    prompt: str = "hello",
) -> RunRecord:
    """Build one new run through the domain factory."""

    return RunRecord.start(
        id=run_id,
        session_id=session_id,
        prompt=prompt,
        model="gpt-5.4",
        provider="test",
    )


def create_session(
    store: SQLiteStore,
    *,
    session_id: str,
    name: str | None = None,
) -> SessionRecord:
    """Create one session through the record-based Store contract."""

    record = SessionRecord.create(id=session_id, name=name)
    return store.create_session(record=record)

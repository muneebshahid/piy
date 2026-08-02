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

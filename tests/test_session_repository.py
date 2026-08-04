"""Tests for store-bound sessions and the session repository."""

from collections.abc import Iterator
from typing import override
from uuid import UUID

import pytest

from tests.support.store import persist_outcome, start_run
from tile import Completed, RunRecord
from tile.sessions import SessionRepository
from tile.store import HistoryItem, SQLiteStore
from tile.types import AssistantTurn, UserMessage


class _ObservedStore(SQLiteStore):
    """SQLite Store that counts standalone scoped collection reads."""

    def __init__(self) -> None:
        """Create an in-memory observed Store."""

        super().__init__(in_memory=True)
        self.history_reads = 0
        self.run_reads = 0

    @override
    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Count and delegate one public history read."""

        self.history_reads += 1
        return super().get_history(session_id)

    @override
    def list_runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """Count and delegate one public run collection read."""

        self.run_reads += 1
        return super().list_runs(session_id)


@pytest.fixture
def observed_store() -> Iterator[_ObservedStore]:
    """Yield a read-counting in-memory Store closed on teardown."""

    store = _ObservedStore()
    try:
        yield store
    finally:
        store.close()


def test_repository_creates_gets_and_lists_lightweight_sessions(
    observed_store: _ObservedStore,
) -> None:
    """List store-bound handles without loading their histories or runs."""

    repository = SessionRepository(observed_store)
    generated = repository.create()
    explicit = repository.create(session_id="known")

    fetched = repository.get("known")
    listed = repository.list()

    assert generated.id != explicit.id
    assert fetched.id == "known"
    assert [session.id for session in listed] == [generated.id, "known"]
    assert observed_store.history_reads == 0
    assert observed_store.run_reads == 0


def test_session_reads_its_current_record_history_and_runs(
    store: SQLiteStore,
) -> None:
    """Resolve scoped durable data only when the caller requests it."""

    repository = SessionRepository(store)
    session = repository.create(session_id="session-1")
    started = start_run(store)
    history = (
        UserMessage(content="hello"),
        AssistantTurn(response_id="response-1"),
    )
    persist_outcome(store, outcome=Completed(value="done"), history_delta=history)

    assert session.get_session_record().id == "session-1"
    assert session.get_history() == history
    assert session.get_runs() == (
        store.get_run(session_id=session.id, run_id=started.run.id),
    )


def test_repository_forks_committed_history_without_loading_it(
    observed_store: _ObservedStore,
) -> None:
    """Delegate atomic history copying and return the target session handle."""

    repository = SessionRepository(observed_store)
    repository.create(session_id="session-1")
    start_run(observed_store)
    history = (
        UserMessage(content="hello"),
        AssistantTurn(response_id="response-1"),
    )
    persist_outcome(
        observed_store,
        outcome=Completed(value="done"),
        history_delta=history,
    )

    fork = repository.fork(
        "session-1",
        target_session_id="fork",
    )

    assert fork.id == "fork"
    assert fork.get_session_record().id == "fork"
    assert fork.get_history() == history
    assert observed_store.history_reads == 1


def test_repository_fork_generates_a_target_session_id_when_omitted(
    store: SQLiteStore,
) -> None:
    """Fork onto a fresh generated session id when no target id is given."""

    repository = SessionRepository(store)
    repository.create(session_id="source")

    fork = repository.fork("source")

    assert UUID(fork.id).version == 4
    assert fork.get_session_record().id == fork.id

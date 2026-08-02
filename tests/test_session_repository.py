"""Tests for store-bound sessions and the session repository."""

import pytest

from tile import Aborted, Completed, RunNotFoundError, RunRecord, SessionNotFoundError
from tile.sessions import SessionRepository
from tile.store import HistoryItem, SQLiteStore
from tile.types import AssistantTurn, UserMessage
from tests.support.store import persist_outcome, start_run


def test_repository_creates_gets_and_lists_lightweight_sessions() -> None:
    """List store-bound handles without loading their histories or runs."""

    store = _ObservedStore()
    repository = SessionRepository(store)
    generated = repository.create()
    explicit = repository.create(session_id="known")

    fetched = repository.get("known")
    listed = repository.list()

    assert generated.id != explicit.id
    assert fetched.id == "known"
    assert [session.id for session in listed] == [generated.id, "known"]
    assert store.history_reads == 0
    assert store.run_reads == 0
    store.close()


def test_session_reads_its_current_record_history_and_runs() -> None:
    """Resolve scoped durable data only when the caller requests it."""

    store = SQLiteStore(in_memory=True)
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
    store.close()


def test_session_scopes_atomic_run_lifecycle_operations() -> None:
    """Preserve Store bootstrap state and fence records from other sessions."""

    store = SQLiteStore(in_memory=True)
    repository = SessionRepository(store)
    session = repository.create(session_id="session-1")
    started = session._start_run(
        run_id="run-1",
        prompt="hello",
        model="gpt-5.4",
        provider="test",
    )
    finished = session._finish_run(
        started.run.id,
        outcome=Completed(value="done"),
        history_delta=(UserMessage(content="hello"),),
    )

    assert started.run.session_id == session.id
    assert started.run.status == "running"
    assert started.committed_history == ()
    assert session._get_run(started.run.id) == finished
    store.close()


def test_repository_durably_aborts_the_active_run_by_session_id() -> None:
    """Expose an idempotent escape hatch without implying task cancellation."""

    store = SQLiteStore(in_memory=True)
    repository = SessionRepository(store)
    repository.create(session_id="session-1")
    started = start_run(store)

    aborted = repository.abort_active_run("session-1")

    assert aborted is not None
    assert aborted.id == started.run.id
    assert aborted.outcome == Aborted(reason="cancelled")
    assert repository.abort_active_run("session-1") is None
    store.close()


def test_repository_forks_committed_history_without_loading_it() -> None:
    """Delegate atomic history copying and return the target session handle."""

    store = _ObservedStore()
    repository = SessionRepository(store)
    repository.create(session_id="session-1")
    start_run(store)
    history = (
        UserMessage(content="hello"),
        AssistantTurn(response_id="response-1"),
    )
    persist_outcome(store, outcome=Completed(value="done"), history_delta=history)

    fork = repository.fork(
        "session-1",
        target_session_id="fork",
    )

    assert fork.id == "fork"
    assert fork.get_session_record().id == "fork"
    assert fork.get_history() == history
    assert store.history_reads == 1
    store.close()


def test_repository_deletes_a_session_and_all_owned_data() -> None:
    """Delete session metadata, run records, and committed history atomically."""

    store = SQLiteStore(in_memory=True)
    repository = SessionRepository(store)
    repository.create(session_id="session-1")
    started = start_run(store)
    persist_outcome(
        store,
        outcome=Completed(value="done"),
        history_delta=(UserMessage(content="hello"),),
    )

    repository.delete("session-1")

    with pytest.raises(SessionNotFoundError):
        repository.get("session-1")
    with pytest.raises(RunNotFoundError):
        store.get_run(session_id="session-1", run_id=started.run.id)
    store.close()


class _ObservedStore(SQLiteStore):
    """SQLite Store that counts standalone scoped collection reads."""

    def __init__(self) -> None:
        """Create an in-memory observed Store."""

        super().__init__(in_memory=True)
        self.history_reads = 0
        self.run_reads = 0

    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Count and delegate one public history read."""

        self.history_reads += 1
        return super().get_history(session_id)

    def list_runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """Count and delegate one public run collection read."""

        self.run_reads += 1
        return super().list_runs(session_id)

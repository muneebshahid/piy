"""Lifecycle and transaction tests for the unified SQLite Store."""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from tile import (
    Aborted,
    ActiveRunError,
    AgentFailure,
    Completed,
    Failed,
    RunAlreadyExistsError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    RunAlreadyEndedError,
    StorePersistenceError,
)
from tile.store import SQLiteStore
from tile.types import AssistantTurn, UserMessage
from tests.support.store import (
    create_session,
    persist_outcome,
    run_record,
    running_record,
    start_run,
)


def _invoke_start_run(store: SQLiteStore) -> None:
    """Invoke start_run for backend-error translation coverage."""

    store.start_run(record=running_record())


def _invoke_finish_run(store: SQLiteStore) -> None:
    """Invoke finish_run for backend-error translation coverage."""

    store.finish_run(
        run_id="run-1",
        outcome=Completed(value="done"),
        history_delta=(),
    )


def _invoke_abort_active_run(store: SQLiteStore) -> None:
    """Invoke abort_active_run for backend-error translation coverage."""

    store.abort_active_run("session-1")


def _invoke_get_run(store: SQLiteStore) -> None:
    """Invoke get_run for backend-error translation coverage."""

    store.get_run("run-1")


def _invoke_delete_session(store: SQLiteStore) -> None:
    """Invoke delete_session for backend-error translation coverage."""

    store.delete_session("session-1")


def test_sqlite_store_requires_an_explicit_storage_mode() -> None:
    """Require a path unless process-local SQLite was requested."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteStore()


def test_sqlite_store_round_trips_sessions_runs_and_typed_history() -> None:
    """Persist all aggregate records through the unified adapter."""

    store = SQLiteStore(in_memory=True)
    try:
        session = create_session(store, session_id="session-1", name="First")
        started = start_run(store)
        assert started.committed_history == ()
        finished = store.finish_run(
            run_id=started.run.id,
            outcome=Completed(value="done"),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
        )

        stored_session = store.get_session(session.id)
        assert stored_session.id == session.id
        assert stored_session.name == session.name
        assert stored_session.updated_at >= session.updated_at
        assert store.list_sessions() == (stored_session,)
        assert store.get_run(finished.id) == finished
        assert store.list_runs(session.id) == (finished,)
        history = store.get_history(session.id)
        assert [type(item.item) for item in history] == [
            UserMessage,
            AssistantTurn,
        ]
        assert [item.position for item in history] == [0, 1]
        assert {item.run_id for item in history} == {finished.id}
    finally:
        store.close()


def test_sqlite_store_returns_defensive_typed_snapshots() -> None:
    """Prevent nested caller mutation from changing authoritative history."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="original")],
        )
        fetched = store.get_history("session-1")[0].item
        assert isinstance(fetched, UserMessage)
        fetched.content = "mutated"

        stored = store.get_history("session-1")[0].item
        assert isinstance(stored, UserMessage)
        assert stored.content == "original"
    finally:
        store.close()


def test_sqlite_store_rejects_duplicate_and_missing_aggregates() -> None:
    """Translate uniqueness and lookup failures into domain errors."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        with pytest.raises(SessionAlreadyExistsError, match="session-1"):
            create_session(store, session_id="session-1")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.get_session("missing")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.delete_session("missing")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.abort_active_run("missing")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.start_run(
                record=run_record(run_id="run-1", session_id="missing"),
            )

        active = start_run(store).run
        store.abort_active_run("session-1")
        with pytest.raises(RunAlreadyExistsError, match="run-1"):
            store.start_run(
                record=run_record(
                    run_id="run-1",
                    session_id="session-1",
                    prompt="again",
                ),
            )

        assert store.get_run("run-1").id == active.id
        assert store.get_run("run-1").outcome == Aborted(reason="cancelled")
    finally:
        store.close()


def test_start_run_enforces_one_active_run_per_session() -> None:
    """Reject overlap through the store instead of process-local ownership."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        start_run(store)

        with pytest.raises(ActiveRunError, match="session-1"):
            store.start_run(
                record=run_record(
                    run_id="run-2",
                    session_id="session-1",
                    prompt="again",
                ),
            )

        assert [run.id for run in store.list_runs("session-1")] == ["run-1"]
    finally:
        store.close()


def test_start_run_rejects_terminal_record_before_checking_active() -> None:
    """Preserve the active run when the submitted record is already terminal."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        active = start_run(store).run
        terminal = run_record(
            run_id="run-2",
            session_id="session-1",
            prompt="invalid replacement",
        ).finish(outcome=Completed(value="already done"))

        with pytest.raises(ValueError, match="requires a running RunRecord"):
            store.start_run(record=terminal)

        assert store.get_run(active.id) == active
        assert store.list_runs("session-1") == (active,)
    finally:
        store.close()


def test_abort_active_run_finishes_the_record_and_fences_late_writes() -> None:
    """Durably abort a record and reject finalization by its old process."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        active = start_run(store).run

        aborted = store.abort_active_run("session-1")

        assert aborted is not None
        assert aborted.id == active.id
        assert aborted.status == "aborted"
        assert aborted.outcome == Aborted(reason="cancelled")
        with pytest.raises(RunAlreadyEndedError, match="run-1"):
            store.finish_run(
                run_id=active.id,
                outcome=Completed(value="late"),
                history_delta=[UserMessage(content="must not commit")],
            )
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_abort_active_run_is_idempotent_when_no_run_is_running() -> None:
    """Return no record after completion and allow the next normal start."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        completed = persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="hello")],
        )

        aborted = store.abort_active_run("session-1")
        started = store.start_run(
            record=run_record(
                run_id="run-2",
                session_id="session-1",
                prompt="next",
            ),
        )

        assert aborted is None
        assert tuple(item.item for item in started.committed_history) == (
            UserMessage(content="hello"),
        )
        assert store.get_run("run-1") == completed
        assert store.get_run("run-2").status == "running"
    finally:
        store.close()


def test_finish_run_rolls_back_status_when_history_insert_fails(
    tmp_path: Path,
) -> None:
    """Keep the run active when any part of finalization cannot commit."""

    database_path = tmp_path / "failed-history-insert.db"
    seed = SQLiteStore(database_path)
    create_session(seed, session_id="session-1")
    start_run(seed)
    seed.close()
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TRIGGER reject_history_insert
        BEFORE INSERT ON history_items
        BEGIN
            SELECT RAISE(ABORT, 'history insert failed');
        END
        """
    )
    connection.close()

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError) as raised:
            persist_outcome(
                store,
                outcome=Completed(value="done"),
                history_delta=[UserMessage(content="hello")],
            )

        assert raised.value.operation == "finish_run"
        assert isinstance(raised.value.cause, sqlite3.IntegrityError)
        assert "history insert failed" in str(raised.value.cause)
        assert store.get_run("run-1").status == "running"
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_finish_run_rejects_a_second_terminal_transition() -> None:
    """Fence duplicate finalization even when the outcome is identical."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        persist_outcome(
            store,
            outcome=Failed(cause=AgentFailure(reason="cannot deliver")),
            history_delta=[UserMessage(content="hello")],
        )

        with pytest.raises(RunAlreadyEndedError, match="run-1"):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="rewritten"),
                history_delta=[],
            )
        assert store.get_run("run-1").status == "failed"
    finally:
        store.close()


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        pytest.param("start_run", _invoke_start_run, id="start-run"),
        pytest.param("finish_run", _invoke_finish_run, id="finish-run"),
        pytest.param(
            "abort_active_run",
            _invoke_abort_active_run,
            id="abort-active-run",
        ),
        pytest.param("get_run", _invoke_get_run, id="get-run"),
        pytest.param("delete_session", _invoke_delete_session, id="delete-session"),
    ],
)
def test_sqlite_store_translates_backend_errors(
    operation: str,
    invoke: Callable[[SQLiteStore], None],
) -> None:
    """Expose adapter failures through the Store-owned error contract."""

    store = SQLiteStore(in_memory=True)
    store.close()

    with pytest.raises(StorePersistenceError) as raised:
        invoke(store)

    assert raised.value.operation == operation
    assert isinstance(raised.value.cause, sqlite3.ProgrammingError)


def test_delete_session_rolls_back_when_owned_data_deletion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the complete session aggregate when atomic deletion fails."""

    store = SQLiteStore(in_memory=True)
    create_session(store, session_id="session-1")
    start_run(store)
    persist_outcome(
        store,
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
    )

    def fail_run_deletion(session_id: str) -> None:
        """Fail after history deletion to exercise transaction rollback."""

        _ = session_id
        raise sqlite3.OperationalError("cannot delete runs")

    monkeypatch.setattr(store, "_delete_session_runs", fail_run_deletion)

    with pytest.raises(StorePersistenceError) as raised:
        store.delete_session("session-1")

    assert raised.value.operation == "delete_session"
    assert store.get_session("session-1").id == "session-1"
    assert len(store.list_runs("session-1")) == 1
    assert len(store.get_history("session-1")) == 1
    store.close()


def test_file_backed_store_survives_restart(tmp_path: Path) -> None:
    """Reload authoritative state from a reopened SQLite adapter."""

    database_path = tmp_path / "tile.db"
    first = SQLiteStore(database_path)
    create_session(first, session_id="session-1")
    start_run(first)
    persist_outcome(
        first,
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
    )
    first.close()

    reopened = SQLiteStore(database_path)
    try:
        assert reopened.get_run("run-1").status == "completed"
        assert reopened.get_history("session-1")[0].item == UserMessage(content="hello")
    finally:
        reopened.close()


def test_store_translates_corrupted_run_outcome_payloads(tmp_path: Path) -> None:
    """Hide adapter validation details behind operation-specific Store errors."""

    database_path = tmp_path / "invalid-run-outcome.db"
    seed = SQLiteStore(database_path)
    create_session(seed, session_id="session-1")
    start_run(seed)
    persist_outcome(seed, outcome=Completed(value="done"), history_delta=[])
    seed.close()
    connection = sqlite3.connect(database_path)
    connection.execute("UPDATE runs SET outcome_json = 'not-json'")
    connection.commit()
    connection.close()

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError) as get_error:
            store.get_run("run-1")
        assert get_error.value.operation == "get_run"
        assert isinstance(get_error.value.cause, ValidationError)

        with pytest.raises(StorePersistenceError) as list_error:
            store.list_runs("session-1")
        assert list_error.value.operation == "list_runs"
        assert isinstance(list_error.value.cause, ValidationError)
    finally:
        store.close()


def test_start_run_rolls_back_insert_when_history_snapshot_fails(
    tmp_path: Path,
) -> None:
    """Avoid inserting a run when bootstrap history cannot be decoded."""

    database_path = tmp_path / "invalid-bootstrap-history.db"
    seed = SQLiteStore(database_path)
    create_session(seed, session_id="session-1")
    first = start_run(seed)
    seed.finish_run(
        run_id=first.run.id,
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
    )
    seed.close()
    connection = sqlite3.connect(database_path)
    connection.execute("UPDATE history_items SET payload_json = 'not-json'")
    connection.commit()
    connection.close()

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError) as history_error:
            store.get_history("session-1")
        assert history_error.value.operation == "get_history"
        assert isinstance(history_error.value.cause, ValidationError)

        with pytest.raises(StorePersistenceError) as start_error:
            store.start_run(
                record=run_record(
                    run_id="run-2",
                    session_id="session-1",
                    prompt="next",
                ),
            )
        assert start_error.value.operation == "start_run"
        assert isinstance(start_error.value.cause, ValidationError)

        assert [run.id for run in store.list_runs("session-1")] == ["run-1"]
    finally:
        store.close()

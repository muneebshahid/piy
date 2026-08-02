"""Lifecycle and transaction tests for the unified SQLite Store."""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.support.store import (
    corrupt_column,
    create_session,
    persist_outcome,
    seed_database,
    start_run,
)
from tile import (
    Aborted,
    Completed,
    RunAlreadyEndedError,
    RunAlreadyExistsError,
    RunNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StorePersistenceError,
)
from tile.store import SQLiteStore
from tile.types import AssistantTurn, UserMessage


def _invoke_finish_run(store: SQLiteStore) -> None:
    """Invoke finish_run against the default run identifiers."""

    persist_outcome(store, outcome=Completed(value="done"), history_delta=())


def _invoke_abort_active_run(store: SQLiteStore) -> None:
    """Invoke abort_active_run against the default session."""

    store.abort_active_run("session-1")


def _invoke_get_run(store: SQLiteStore) -> None:
    """Invoke get_run against the default run identifiers."""

    store.get_run(session_id="session-1", run_id="run-1")


def _invoke_list_runs(store: SQLiteStore) -> None:
    """Invoke list_runs against the default session."""

    store.list_runs("session-1")


def _invoke_get_history(store: SQLiteStore) -> None:
    """Invoke get_history against the default session."""

    store.get_history("session-1")


def _invoke_start_next_run(store: SQLiteStore) -> None:
    """Invoke start_run for a second run in the default session."""

    start_run(store, run_id="run-2", prompt="next")


def _seed_finished_run(database_path: Path) -> None:
    """Seed one finished run with committed history, then close the store."""

    store = SQLiteStore(database_path)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="hello")],
        )
    finally:
        store.close()


def test_sqlite_store_requires_an_explicit_storage_mode() -> None:
    """Require a path unless process-local SQLite was requested."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteStore()


def test_sqlite_store_round_trips_sessions_runs_and_typed_history(
    store: SQLiteStore,
) -> None:
    """Persist all aggregate records through the unified adapter."""

    session = create_session(store, session_id="session-1")
    started = start_run(store)
    assert started.committed_history == ()
    finished = store.finish_run(
        session_id=session.id,
        run_id=started.run.id,
        outcome=Completed(value="done"),
        history_delta=[
            UserMessage(content="hello"),
            AssistantTurn(response_id="response-1"),
        ],
    )

    stored_session = store.get_session(session.id)
    assert stored_session.id == session.id
    assert stored_session.updated_at >= session.updated_at
    assert store.list_sessions() == (stored_session,)
    assert store.get_run(session_id=session.id, run_id=finished.id) == finished
    assert store.list_runs(session.id) == (finished,)
    history = store.get_history(session.id)
    assert [type(item.item) for item in history] == [
        UserMessage,
        AssistantTurn,
    ]
    assert [item.position for item in history] == [0, 1]
    assert {item.run_id for item in history} == {finished.id}


def test_sqlite_store_returns_defensive_typed_snapshots(store: SQLiteStore) -> None:
    """Prevent nested caller mutation from changing authoritative history."""

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


def test_finish_run_copies_the_caller_owned_outcome(store: SQLiteStore) -> None:
    """Keep nested input mutations out of the returned and durable snapshots."""

    create_session(store, session_id="session-1")
    start_run(store)
    outcome = Completed(value={"nested": {"value": "original"}})

    finished = persist_outcome(store, outcome=outcome, history_delta=())
    assert isinstance(outcome.value, dict)
    nested = outcome.value["nested"]
    assert isinstance(nested, dict)
    nested["value"] = "mutated"

    expected = Completed(value={"nested": {"value": "original"}})
    assert finished.outcome == expected
    assert store.get_run(session_id="session-1", run_id="run-1").outcome == expected


def test_sqlite_store_rejects_duplicate_and_missing_aggregates(
    store: SQLiteStore,
) -> None:
    """Translate uniqueness and lookup failures into domain errors."""

    create_session(store, session_id="session-1")
    with pytest.raises(SessionAlreadyExistsError, match="session-1"):
        create_session(store, session_id="session-1")
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.get_session("missing")
    with pytest.raises(SessionNotFoundError, match="missing"):
        store.abort_active_run("missing")
    with pytest.raises(SessionNotFoundError, match="missing"):
        start_run(store, session_id="missing")

    active = start_run(store).run
    store.abort_active_run("session-1")
    with pytest.raises(RunAlreadyExistsError, match="run-1"):
        start_run(store, prompt="again")

    assert store.get_run(session_id="session-1", run_id="run-1").id == active.id
    assert store.get_run(
        session_id="session-1",
        run_id="run-1",
    ).outcome == Aborted(reason="cancelled")


def test_get_and_finish_run_require_the_owning_session(store: SQLiteStore) -> None:
    """Keep run reads and transitions inside their owning session boundary."""

    create_session(store, session_id="session-1")
    create_session(store, session_id="session-2")
    active = start_run(store, session_id="session-1").run

    with pytest.raises(RunNotFoundError, match="run-1"):
        store.get_run(session_id="session-2", run_id=active.id)
    with pytest.raises(RunNotFoundError, match="run-1"):
        persist_outcome(
            store,
            session_id="session-2",
            run_id=active.id,
            outcome=Completed(value="wrong session"),
            history_delta=[UserMessage(content="must not commit")],
        )

    assert store.get_run(session_id="session-1", run_id=active.id) == active
    assert store.get_history("session-1") == ()
    assert store.get_history("session-2") == ()


def test_abort_active_run_finishes_the_record_and_fences_late_writes(
    store: SQLiteStore,
) -> None:
    """Durably abort a record and reject finalization by its old process."""

    create_session(store, session_id="session-1")
    active = start_run(store).run

    aborted = store.abort_active_run("session-1")

    assert aborted is not None
    assert aborted.id == active.id
    assert aborted.status == "aborted"
    assert aborted.outcome == Aborted(reason="cancelled")
    with pytest.raises(RunAlreadyEndedError, match="run-1"):
        persist_outcome(
            store,
            run_id=active.id,
            outcome=Completed(value="late"),
            history_delta=[UserMessage(content="must not commit")],
        )
    assert store.get_history("session-1") == ()


def test_abort_active_run_is_idempotent_when_no_run_is_running(
    store: SQLiteStore,
) -> None:
    """Return no record after completion and allow the next normal start."""

    create_session(store, session_id="session-1")
    start_run(store)
    completed = persist_outcome(
        store,
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
    )

    aborted = store.abort_active_run("session-1")
    started = start_run(store, run_id="run-2", prompt="next")

    assert aborted is None
    assert tuple(item.item for item in started.committed_history) == (
        UserMessage(content="hello"),
    )
    assert store.get_run(session_id="session-1", run_id="run-1") == completed
    assert store.get_run(session_id="session-1", run_id="run-2").status == "running"


def test_fork_session_copies_all_history_with_new_envelopes(
    store: SQLiteStore,
) -> None:
    """Copy full history while preserving payload and originating run ids."""

    create_session(store, session_id="source")
    start_run(store, session_id="source")
    persist_outcome(
        store,
        session_id="source",
        outcome=Completed(value="done"),
        history_delta=[
            UserMessage(content="hello"),
            AssistantTurn(response_id="response-1"),
        ],
    )

    fork = store.fork_session(
        source_session_id="source",
        target_session_id="fork",
    )

    source = store.get_history("source")
    copied = store.get_history(fork.id)
    assert len(copied) == len(source) == 2
    assert [item.id for item in copied] != [item.id for item in source]
    assert {item.session_id for item in copied} == {"fork"}
    assert [item.run_id for item in copied] == [item.run_id for item in source]
    assert [item.position for item in copied] == [item.position for item in source]
    assert [item.item for item in copied] == [item.item for item in source]
    assert [item.created_at for item in copied] == [item.created_at for item in source]
    assert store.list_runs("fork") == ()


def test_fork_session_requires_an_existing_source_session(store: SQLiteStore) -> None:
    """Reject forking from a session id that was never created."""

    with pytest.raises(SessionNotFoundError, match="missing"):
        store.fork_session(source_session_id="missing", target_session_id="fork")


def test_fork_session_rejects_an_existing_target_session(store: SQLiteStore) -> None:
    """Reject forking onto a target session id that already exists."""

    create_session(store, session_id="source")
    create_session(store, session_id="target")

    with pytest.raises(SessionAlreadyExistsError, match="target"):
        store.fork_session(source_session_id="source", target_session_id="target")


def test_finish_run_rolls_back_status_when_history_insert_fails(
    tmp_path: Path,
) -> None:
    """Keep the run active when any part of finalization cannot commit."""

    database_path = tmp_path / "failed-history-insert.db"
    seed_database(database_path, running_run=True)
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
        assert store.get_run(session_id="session-1", run_id="run-1").status == "running"
        assert store.get_history("session-1") == ()
    finally:
        store.close()


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        pytest.param("start_run", start_run, id="start-run"),
        pytest.param("finish_run", _invoke_finish_run, id="finish-run"),
        pytest.param(
            "abort_active_run",
            _invoke_abort_active_run,
            id="abort-active-run",
        ),
        pytest.param("get_run", _invoke_get_run, id="get-run"),
    ],
)
def test_sqlite_store_translates_backend_errors(
    operation: str,
    invoke: Callable[[SQLiteStore], object],
    store: SQLiteStore,
) -> None:
    """Expose adapter failures through the Store-owned error contract."""

    store.close()

    with pytest.raises(StorePersistenceError) as raised:
        invoke(store)

    assert raised.value.operation == operation
    assert isinstance(raised.value.cause, sqlite3.ProgrammingError)


def test_file_backed_store_survives_restart(tmp_path: Path) -> None:
    """Reload authoritative state from a reopened SQLite adapter."""

    database_path = tmp_path / "tile.db"
    _seed_finished_run(database_path)

    reopened = SQLiteStore(database_path)
    try:
        assert (
            reopened.get_run(session_id="session-1", run_id="run-1").status
            == "completed"
        )
        assert reopened.get_history("session-1")[0].item == UserMessage(content="hello")
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("table", "column", "operation", "invoke"),
    [
        pytest.param(
            "runs",
            "outcome_json",
            "get_run",
            _invoke_get_run,
            id="run-outcome-get-run",
        ),
        pytest.param(
            "runs",
            "outcome_json",
            "list_runs",
            _invoke_list_runs,
            id="run-outcome-list-runs",
        ),
        pytest.param(
            "history_items",
            "payload_json",
            "get_history",
            _invoke_get_history,
            id="history-payload-get-history",
        ),
        pytest.param(
            "history_items",
            "payload_json",
            "start_run",
            _invoke_start_next_run,
            id="history-payload-start-run",
        ),
    ],
)
def test_store_translates_corrupted_persistent_payloads(
    tmp_path: Path,
    table: str,
    column: str,
    operation: str,
    invoke: Callable[[SQLiteStore], object],
) -> None:
    """Hide adapter validation details behind operation-specific Store errors."""

    database_path = tmp_path / "corrupted-payload.db"
    _seed_finished_run(database_path)
    corrupt_column(database_path, table=table, column=column)

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError) as raised:
            invoke(store)
    finally:
        store.close()

    assert raised.value.operation == operation
    assert isinstance(raised.value.cause, ValidationError)


def test_start_run_rolls_back_insert_when_history_snapshot_fails(
    tmp_path: Path,
) -> None:
    """Avoid inserting a run when bootstrap history cannot be decoded."""

    database_path = tmp_path / "invalid-bootstrap-history.db"
    _seed_finished_run(database_path)
    corrupt_column(database_path, table="history_items", column="payload_json")

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError):
            start_run(store, run_id="run-2", prompt="next")

        assert [run.id for run in store.list_runs("session-1")] == ["run-1"]
    finally:
        store.close()

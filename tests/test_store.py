"""Contract and transaction tests for the unified SQLite Store."""

import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from tile import (
    Aborted,
    ActiveRunError,
    AgentFailure,
    Completed,
    Failed,
    RunAlreadyExistsError,
    RunRecord,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StaleRunError,
    StorePersistenceError,
)
from tile.store import (
    SQLiteStore,
    SQLiteStoreSchemaError,
    SessionRecord,
    StartedRun,
)
from tile.types import (
    AssistantTurn,
    ConversationItem,
    UserMessage,
)

STARTED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def _invoke_start_run(store: SQLiteStore) -> None:
    """Invoke start_run for backend-error translation coverage."""

    store.start_run(
        record=_running_record(),
    )


def _invoke_finish_run(store: SQLiteStore) -> None:
    """Invoke finish_run for backend-error translation coverage."""

    store.finish_run(
        record=_running_record().finish(outcome=Completed(value="done")),
        history_delta=(),
    )


def _invoke_get_run(store: SQLiteStore) -> None:
    """Invoke get_run for backend-error translation coverage."""

    store.get_run("run-1")


def test_sqlite_store_requires_an_explicit_storage_mode() -> None:
    """Require a path unless process-local SQLite was requested."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteStore()


def test_sqlite_store_round_trips_sessions_runs_and_typed_history() -> None:
    """Persist all aggregate records through the unified adapter."""

    store = SQLiteStore(in_memory=True)
    try:
        session = _create_session(store, session_id="session-1", name="First")
        started = _start_run(store)
        assert started.committed_history == ()
        finished = store.finish_run(
            record=started.run.finish(outcome=Completed(value="done")),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
        )

        stored_session = store.get_session(session.session_id)
        assert stored_session.session_id == session.session_id
        assert stored_session.name == session.name
        assert stored_session.updated_at >= session.updated_at
        assert store.list_sessions() == (stored_session,)
        assert store.get_run(finished.run_id) == finished
        assert store.list_runs(session.session_id) == (finished,)
        history = store.get_history(session.session_id)
        assert [type(item.item) for item in history] == [
            UserMessage,
            AssistantTurn,
        ]
        assert [item.position for item in history] == [0, 1]
        assert {item.run_id for item in history} == {finished.run_id}
    finally:
        store.close()


def test_sqlite_store_returns_defensive_typed_snapshots() -> None:
    """Prevent nested caller mutation from changing authoritative history."""

    store = SQLiteStore(in_memory=True)
    try:
        _create_session(store, session_id="session-1")
        _start_run(store)
        _persist_outcome(
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
        _create_session(store, session_id="session-1")
        with pytest.raises(SessionAlreadyExistsError, match="session-1"):
            _create_session(store, session_id="session-1")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.get_session("missing")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.start_run(
                record=_run_record(run_id="run-1", session_id="missing"),
            )

        _start_run(store)
        with pytest.raises(RunAlreadyExistsError, match="run-1"):
            store.start_run(
                record=_run_record(
                    run_id="run-1",
                    session_id="session-1",
                    prompt="again",
                ),
                replace_active=True,
            )
    finally:
        store.close()


def test_start_run_enforces_one_active_run_per_session() -> None:
    """Reject overlap through the store instead of process-local ownership."""

    store = SQLiteStore(in_memory=True)
    try:
        _create_session(store, session_id="session-1")
        _start_run(store)

        with pytest.raises(ActiveRunError, match="session-1"):
            store.start_run(
                record=_run_record(
                    run_id="run-2",
                    session_id="session-1",
                    prompt="again",
                ),
            )

        assert [run.run_id for run in store.list_runs("session-1")] == ["run-1"]
    finally:
        store.close()


def test_replace_active_finishes_old_run_and_fences_late_writes() -> None:
    """Replace and create in one transaction, then reject old finalization."""

    store = SQLiteStore(in_memory=True)
    try:
        _create_session(store, session_id="session-1")
        first = _start_run(store)
        late = first.run.finish(outcome=Completed(value="late"))

        second = store.start_run(
            record=_run_record(
                run_id="run-2",
                session_id="session-1",
                prompt="replacement",
            ),
            replace_active=True,
        )

        assert second.replaced_run_id == first.run.run_id
        replaced = store.get_run(first.run.run_id)
        assert replaced.status == "aborted"
        assert replaced.outcome == Aborted(reason="replaced")
        with pytest.raises(StaleRunError, match="run-1"):
            store.finish_run(
                record=late,
                history_delta=[UserMessage(content="must not commit")],
            )
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_replace_active_does_not_rewrite_an_already_finished_run() -> None:
    """Start normally when the prior process committed before replacement."""

    store = SQLiteStore(in_memory=True)
    try:
        _create_session(store, session_id="session-1")
        _start_run(store)
        completed = _persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="hello")],
        )

        started = store.start_run(
            record=_run_record(
                run_id="run-2",
                session_id="session-1",
                prompt="next",
            ),
            replace_active=True,
        )

        assert started.replaced_run_id is None
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
    _create_session(seed, session_id="session-1")
    _start_run(seed)
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
            _persist_outcome(
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
        _create_session(store, session_id="session-1")
        started = _start_run(store)
        rewritten = started.run.finish(outcome=Completed(value="rewritten"))
        _persist_outcome(
            store,
            outcome=Failed(cause=AgentFailure(reason="cannot deliver")),
            history_delta=[UserMessage(content="hello")],
        )

        with pytest.raises(StaleRunError, match="run-1"):
            store.finish_run(
                record=rewritten,
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
        pytest.param("get_run", _invoke_get_run, id="get-run"),
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


def test_fork_session_copies_all_history_with_new_envelopes() -> None:
    """Copy full history while preserving payload and originating run ids."""

    store = SQLiteStore(in_memory=True)
    try:
        _create_session(store, session_id="source")
        _start_run(store, session_id="source")
        _persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
        )

        fork = store.fork_session(
            source_session_id="source",
            target=SessionRecord.create(session_id="fork", name="Fork"),
        )

        source = store.get_history("source")
        copied = store.get_history(fork.session_id)
        assert len(copied) == len(source) == 2
        assert [item.id for item in copied] != [item.id for item in source]
        assert {item.session_id for item in copied} == {"fork"}
        assert [item.run_id for item in copied] == [item.run_id for item in source]
        assert [item.position for item in copied] == [item.position for item in source]
        assert [item.item for item in copied] == [item.item for item in source]
        assert [item.created_at for item in copied] == [
            item.created_at for item in source
        ]
        assert store.list_runs("fork") == ()
    finally:
        store.close()


def test_file_backed_store_survives_restart(tmp_path: Path) -> None:
    """Reload authoritative state from a reopened SQLite adapter."""

    database_path = tmp_path / "tile.db"
    first = SQLiteStore(database_path)
    _create_session(first, session_id="session-1")
    _start_run(first)
    _persist_outcome(
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


def test_start_run_rolls_back_replacement_when_history_snapshot_fails(
    tmp_path: Path,
) -> None:
    """Keep the predecessor active when bootstrap history cannot be decoded."""

    database_path = tmp_path / "invalid-bootstrap-history.db"
    seed = SQLiteStore(database_path)
    _create_session(seed, session_id="session-1")
    first = _start_run(seed)
    seed.finish_run(
        record=first.run.finish(outcome=Completed(value="done")),
        history_delta=[UserMessage(content="hello")],
    )
    active = seed.start_run(
        record=_run_record(
            run_id="run-2",
            session_id="session-1",
            prompt="active",
        ),
    ).run
    seed.close()
    connection = sqlite3.connect(database_path)
    connection.execute("UPDATE history_items SET payload_json = 'not-json'")
    connection.commit()
    connection.close()

    store = SQLiteStore(database_path)
    try:
        with pytest.raises(ValidationError):
            store.start_run(
                record=_run_record(
                    run_id="run-3",
                    session_id="session-1",
                    prompt="replacement",
                ),
                replace_active=True,
            )

        assert store.get_run("run-2") == active
        assert [run.run_id for run in store.list_runs("session-1")] == [
            "run-1",
            "run-2",
        ]
    finally:
        store.close()


def test_unified_store_rejects_legacy_schema_without_migration(
    tmp_path: Path,
) -> None:
    """Fail clearly rather than reinterpret split-store development data."""

    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('schema_version', '2')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match="pre-unified"):
        SQLiteStore(database_path)


def test_unified_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Reject data written by an unsupported future unified schema."""

    database_path = tmp_path / "future.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('store_schema_version', '999')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match="999"):
        SQLiteStore(database_path)


def test_unified_schema_declares_required_foreign_keys(tmp_path: Path) -> None:
    """Tie runs and history to their persistent aggregate records."""

    database_path = tmp_path / "constraints.db"
    store = SQLiteStore(database_path)
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        run_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(runs)"
        ).fetchall()
        history_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(history_items)"
        ).fetchall()
    finally:
        connection.close()

    assert {(row[2], row[3], row[4]) for row in run_foreign_keys} == {
        ("sessions", "session_id", "id")
    }
    assert {(row[2], row[3], row[4]) for row in history_foreign_keys} == {
        ("runs", "run_id", "id"),
        ("sessions", "session_id", "id"),
    }


def test_unified_schema_requires_run_provider_identity(tmp_path: Path) -> None:
    """Reject a persistent run whose provider was not known at creation."""

    database_path = tmp_path / "provider-constraint.db"
    store = SQLiteStore(database_path)
    _create_session(store, session_id="session-1")
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, prompt, status, started_at,
                    ended_at, model, provider, outcome_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "invalid",
                    "session-1",
                    "hello",
                    "running",
                    STARTED_AT.isoformat(),
                    None,
                    "gpt-5.4",
                    None,
                    None,
                ),
            )
    finally:
        connection.close()


def test_unified_schema_rejects_inconsistent_run_lifecycle_rows(
    tmp_path: Path,
) -> None:
    """Enforce basic terminal-field agreement below the domain adapter."""

    database_path = tmp_path / "lifecycle-constraint.db"
    store = SQLiteStore(database_path)
    _create_session(store, session_id="session-1")
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, prompt, status, started_at,
                    ended_at, model, provider, outcome_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "invalid",
                    "session-1",
                    "hello",
                    "completed",
                    STARTED_AT.isoformat(),
                    None,
                    "gpt-5.4",
                    "test",
                    None,
                ),
            )
    finally:
        connection.close()


def test_concurrent_starts_leave_exactly_one_running_run(tmp_path: Path) -> None:
    """Serialize competing starts across independent Store instances."""

    database_path = tmp_path / "starts.db"
    seed = SQLiteStore(database_path)
    _create_session(seed, session_id="session-1")
    seed.close()
    barrier = Barrier(2)

    def start(run_id: str) -> str:
        """Race one run start and report its domain result."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                record=_run_record(
                    run_id=run_id,
                    session_id="session-1",
                    prompt=run_id,
                ),
            )
            return "started"
        except ActiveRunError:
            return "busy"
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(start, ("run-1", "run-2")))

    reopened = SQLiteStore(database_path)
    try:
        assert sorted(results) == ["busy", "started"]
        runs = reopened.list_runs("session-1")
        assert sum(run.status == "running" for run in runs) == 1
    finally:
        reopened.close()


def test_concurrent_replacements_leave_one_running_successor(tmp_path: Path) -> None:
    """Serialize replacement attempts without violating active-run uniqueness."""

    database_path = tmp_path / "replacements.db"
    seed = SQLiteStore(database_path)
    _create_session(seed, session_id="session-1")
    _start_run(seed)
    seed.close()
    barrier = Barrier(2)

    def replace(run_id: str) -> str:
        """Race one replacement through an independent Store instance."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                record=_run_record(
                    run_id=run_id,
                    session_id="session-1",
                    prompt=run_id,
                ),
                replace_active=True,
            )
            return run_id
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        replaced_by = list(executor.map(replace, ("run-2", "run-3")))

    reopened = SQLiteStore(database_path)
    try:
        runs = reopened.list_runs("session-1")
        assert set(replaced_by) == {"run-2", "run-3"}
        assert len(runs) == 3
        assert sum(run.status == "running" for run in runs) == 1
        assert sum(run.outcome == Aborted(reason="replaced") for run in runs) == 2
    finally:
        reopened.close()


def test_finish_and_replace_race_preserves_one_valid_winner(tmp_path: Path) -> None:
    """Keep completion or replacement intact according to transaction order."""

    database_path = tmp_path / "finish-replace.db"
    _seed_running_run(database_path)
    barrier = Barrier(2)

    def finish() -> str:
        """Race terminal completion against replacement."""

        store = SQLiteStore(database_path)
        try:
            record = store.get_run("run-1").finish(outcome=Completed(value="done"))
            barrier.wait()
            store.finish_run(
                record=record,
                history_delta=[UserMessage(content="hello")],
            )
            return "finished"
        except StaleRunError:
            return "stale"
        finally:
            store.close()

    def replace() -> str | None:
        """Race replacement against terminal completion."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            started = store.start_run(
                record=_run_record(
                    run_id="run-2",
                    session_id="session-1",
                    prompt="replacement",
                ),
                replace_active=True,
            )
            return started.replaced_run_id
        finally:
            store.close()

    finish_result, replaced_run_id = _run_finish_replace_race(finish, replace)
    _assert_finish_replace_result(
        database_path,
        finish_result=finish_result,
        replaced_run_id=replaced_run_id,
    )


def _seed_running_run(database_path: Path) -> None:
    """Create the shared run used by a finish-versus-replace race."""

    seed = SQLiteStore(database_path)
    try:
        _create_session(seed, session_id="session-1")
        _start_run(seed)
    finally:
        seed.close()


def _run_finish_replace_race(
    finish: Callable[[], str],
    replace: Callable[[], str | None],
) -> tuple[str, str | None]:
    """Run the two lifecycle transactions concurrently."""

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish_future = executor.submit(finish)
        replace_future = executor.submit(replace)
        return finish_future.result(), replace_future.result()


def _assert_finish_replace_result(
    database_path: Path,
    *,
    finish_result: str,
    replaced_run_id: str | None,
) -> None:
    """Assert state produced by either valid transaction ordering."""

    reopened = SQLiteStore(database_path)
    try:
        old = reopened.get_run("run-1")
        assert reopened.get_run("run-2").status == "running"
        if finish_result == "finished":
            assert replaced_run_id is None
            assert old.status == "completed"
            assert len(reopened.get_history("session-1")) == 1
        else:
            assert replaced_run_id == "run-1"
            assert old.outcome == Aborted(reason="replaced")
            assert reopened.get_history("session-1") == ()
    finally:
        reopened.close()


def _start_run(
    store: SQLiteStore,
    *,
    session_id: str = "session-1",
) -> StartedRun:
    """Start one run through the record-based Store contract."""

    return store.start_run(
        record=_run_record(run_id="run-1", session_id=session_id),
    )


def _persist_outcome(
    store: SQLiteStore,
    *,
    outcome: Completed | Failed | Aborted,
    history_delta: Sequence[ConversationItem],
    run_id: str = "run-1",
) -> RunRecord:
    """Finish and persist one Store-owned running record."""

    record = store.get_run(run_id).finish(outcome=outcome)
    return store.finish_run(record=record, history_delta=history_delta)


def _running_record() -> RunRecord:
    """Build one running record without Store access."""

    return _run_record(run_id="run-1", session_id="session-1")


def _run_record(
    *,
    run_id: str,
    session_id: str,
    prompt: str = "hello",
) -> RunRecord:
    """Build one new run through the domain factory."""

    return RunRecord.start(
        run_id=run_id,
        session_id=session_id,
        prompt=prompt,
        model="gpt-5.4",
        provider="test",
    )


def _create_session(
    store: SQLiteStore,
    *,
    session_id: str,
    name: str | None = None,
) -> SessionRecord:
    """Create one session through the record-based Store contract."""

    record = SessionRecord.create(session_id=session_id, name=name)
    return store.create_session(record=record)

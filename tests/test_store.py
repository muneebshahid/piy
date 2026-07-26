"""Contract and transaction tests for the unified SQLite Store."""

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from pydantic import ValidationError

from tile import (
    Aborted,
    ActiveRunError,
    AgentFailure,
    Completed,
    Failed,
    InvalidHistoryError,
    RunAlreadyExistsError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StaleRunError,
)
from tile.store import SQLiteStore, SQLiteStoreSchemaError, StartedRun
from tile.types import (
    AssistantTurn,
    ConversationItem,
    ToolCallBlock,
    ToolResultTurn,
    ToolTextContent,
    UserMessage,
)

STARTED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
ENDED_AT = STARTED_AT + timedelta(seconds=2)


def test_sqlite_store_requires_an_explicit_storage_mode() -> None:
    """Require a path unless process-local SQLite was requested."""

    with pytest.raises(ValueError, match="database_path is required"):
        SQLiteStore()


def test_sqlite_store_round_trips_sessions_runs_and_typed_history() -> None:
    """Persist all aggregate records through the unified adapter."""

    store = SQLiteStore(in_memory=True)
    try:
        session = store.create_session(session_id="session-1", name="First")
        started = _start_run(store)
        assert started.committed_history == ()
        finished = store.finish_run(
            run_id=started.run.run_id,
            outcome=Completed(value="done"),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
            ended_at=ENDED_AT,
        )

        assert store.get_session(session.session_id) == session
        assert store.list_sessions() == (session,)
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
        store.create_session(session_id="session-1")
        _start_run(store)
        store.finish_run(
            run_id="run-1",
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="original")],
            ended_at=ENDED_AT,
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
        store.create_session(session_id="session-1")
        with pytest.raises(SessionAlreadyExistsError, match="session-1"):
            store.create_session(session_id="session-1")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.get_session("missing")
        with pytest.raises(SessionNotFoundError, match="missing"):
            store.start_run(
                run_id="run-1",
                session_id="missing",
                prompt="hello",
                model="gpt-5.4",
                provider="test",
            )

        _start_run(store)
        with pytest.raises(RunAlreadyExistsError, match="run-1"):
            store.start_run(
                run_id="run-1",
                session_id="session-1",
                prompt="again",
                model="gpt-5.4",
                provider="test",
                replace_active=True,
            )
    finally:
        store.close()


def test_start_run_enforces_one_active_run_per_session() -> None:
    """Reject overlap through the store instead of process-local ownership."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)

        with pytest.raises(ActiveRunError, match="session-1"):
            store.start_run(
                run_id="run-2",
                session_id="session-1",
                prompt="again",
                model="gpt-5.4",
                provider="test",
            )

        assert [run.run_id for run in store.list_runs("session-1")] == ["run-1"]
    finally:
        store.close()


def test_replace_active_finishes_old_run_and_fences_late_writes() -> None:
    """Replace and create in one transaction, then reject old finalization."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        first = _start_run(store)

        second = store.start_run(
            run_id="run-2",
            session_id="session-1",
            prompt="replacement",
            model="gpt-5.4",
            provider="test",
            replace_active=True,
            started_at=ENDED_AT,
        )

        assert second.replaced_run_id == first.run.run_id
        replaced = store.get_run(first.run.run_id)
        assert replaced.status == "aborted"
        assert replaced.outcome == Aborted(reason="replaced")
        with pytest.raises(StaleRunError, match="run-1"):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="late"),
                history_delta=[UserMessage(content="must not commit")],
            )
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_replace_active_does_not_rewrite_an_already_finished_run() -> None:
    """Start normally when the prior process committed before replacement."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)
        completed = store.finish_run(
            run_id="run-1",
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="hello")],
            ended_at=ENDED_AT,
        )

        started = store.start_run(
            run_id="run-2",
            session_id="session-1",
            prompt="next",
            model="gpt-5.4",
            provider="test",
            replace_active=True,
            started_at=ENDED_AT + timedelta(seconds=1),
        )

        assert started.replaced_run_id is None
        assert tuple(item.item for item in started.committed_history) == (
            UserMessage(content="hello"),
        )
        assert store.get_run("run-1") == completed
        assert store.get_run("run-2").status == "running"
    finally:
        store.close()


def test_finish_run_rolls_back_status_when_history_insert_fails() -> None:
    """Keep the run active when any part of finalization cannot commit."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)
        invalid_item = cast(
            ConversationItem,
            {"role": "not-a-domain-object"},
        )

        with pytest.raises(InvalidHistoryError):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
                history_delta=[invalid_item],
                ended_at=ENDED_AT,
            )

        assert store.get_run("run-1").status == "running"
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_finish_run_rejects_a_second_terminal_transition() -> None:
    """Fence duplicate finalization even when the outcome is identical."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)
        store.finish_run(
            run_id="run-1",
            outcome=Failed(cause=AgentFailure(reason="cannot deliver")),
            history_delta=[UserMessage(content="hello")],
            ended_at=ENDED_AT,
        )

        with pytest.raises(StaleRunError, match="run-1"):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="rewritten"),
                history_delta=[],
            )
        assert store.get_run("run-1").status == "failed"
    finally:
        store.close()


def test_finish_run_rejects_structurally_invalid_history() -> None:
    """Reject unmatched tool results without changing the running record."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)
        orphan = ToolResultTurn(
            call_id="missing-call",
            tool_name="weather",
            content=[ToolTextContent(text="sunny")],
        )

        with pytest.raises(InvalidHistoryError, match="pending call"):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
                history_delta=[UserMessage(content="hello"), orphan],
            )

        assert store.get_run("run-1").status == "running"
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_finish_run_rejects_an_incomplete_assistant_turn() -> None:
    """Keep failed or aborted assistant turns out of committed replay history."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="session-1")
        _start_run(store)
        incomplete = AssistantTurn(status="error", error_message="provider failed")

        with pytest.raises(InvalidHistoryError, match="must be completed"):
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
                history_delta=[UserMessage(content="hello"), incomplete],
            )

        assert store.get_run("run-1").status == "running"
        assert store.get_history("session-1") == ()
    finally:
        store.close()


def test_fork_session_copies_a_flat_prefix_with_new_envelopes() -> None:
    """Copy flat history while preserving payload and originating run ids."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="source")
        _start_run(store, session_id="source")
        store.finish_run(
            run_id="run-1",
            outcome=Completed(value="done"),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
            ended_at=ENDED_AT,
        )

        fork = store.fork_session(
            source_session_id="source",
            target_session_id="fork",
            name="Fork",
            through_position=0,
        )

        source = store.get_history("source")
        copied = store.get_history(fork.session_id)
        assert len(copied) == 1
        assert copied[0].id != source[0].id
        assert copied[0].session_id == "fork"
        assert copied[0].run_id == source[0].run_id
        assert copied[0].position == source[0].position
        assert copied[0].item == source[0].item
        assert copied[0].created_at == source[0].created_at
        assert store.list_runs("fork") == ()
    finally:
        store.close()


def test_fork_rejects_a_prefix_with_an_unanswered_tool_call() -> None:
    """Roll back a fork whose selected prefix is not independently replayable."""

    store = SQLiteStore(in_memory=True)
    try:
        store.create_session(session_id="source")
        _start_run(store, session_id="source")
        assistant = AssistantTurn(
            blocks=[
                ToolCallBlock(
                    call_id="call-1",
                    name="weather",
                    arguments={},
                )
            ],
            stop_reason="tool_use",
        )
        result = ToolResultTurn(
            call_id="call-1",
            tool_name="weather",
            content=[ToolTextContent(text="sunny")],
        )
        store.finish_run(
            run_id="run-1",
            outcome=Completed(value="done"),
            history_delta=[UserMessage(content="hello"), assistant, result],
            ended_at=ENDED_AT,
        )

        with pytest.raises(InvalidHistoryError, match="Unanswered tool calls"):
            store.fork_session(
                source_session_id="source",
                target_session_id="invalid-fork",
                through_position=1,
            )
        with pytest.raises(SessionNotFoundError, match="invalid-fork"):
            store.get_session("invalid-fork")
    finally:
        store.close()


def test_file_backed_store_survives_restart(tmp_path: Path) -> None:
    """Reload authoritative state from a reopened SQLite adapter."""

    database_path = tmp_path / "tile.db"
    first = SQLiteStore(database_path)
    first.create_session(session_id="session-1")
    _start_run(first)
    first.finish_run(
        run_id="run-1",
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
        ended_at=ENDED_AT,
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
    seed.create_session(session_id="session-1")
    first = _start_run(seed)
    seed.finish_run(
        run_id=first.run.run_id,
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
        ended_at=ENDED_AT,
    )
    active = seed.start_run(
        run_id="run-2",
        session_id="session-1",
        prompt="active",
        model="gpt-5.4",
        provider="test",
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
                run_id="run-3",
                session_id="session-1",
                prompt="replacement",
                model="gpt-5.4",
                provider="test",
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


def test_unified_schema_rejects_inconsistent_run_lifecycle_rows(
    tmp_path: Path,
) -> None:
    """Enforce basic terminal-field agreement below the domain adapter."""

    database_path = tmp_path / "lifecycle-constraint.db"
    store = SQLiteStore(database_path)
    store.create_session(session_id="session-1")
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
    seed.create_session(session_id="session-1")
    seed.close()
    barrier = Barrier(2)

    def start(run_id: str) -> str:
        """Race one run start and report its domain result."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                run_id=run_id,
                session_id="session-1",
                prompt=run_id,
                model="gpt-5.4",
                provider="test",
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
    seed.create_session(session_id="session-1")
    _start_run(seed)
    seed.close()
    barrier = Barrier(2)

    def replace(run_id: str) -> str:
        """Race one replacement through an independent Store instance."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                run_id=run_id,
                session_id="session-1",
                prompt=run_id,
                model="gpt-5.4",
                provider="test",
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
            barrier.wait()
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
                history_delta=[UserMessage(content="hello")],
                ended_at=ENDED_AT,
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
                run_id="run-2",
                session_id="session-1",
                prompt="replacement",
                model="gpt-5.4",
                provider="test",
                replace_active=True,
                started_at=ENDED_AT + timedelta(seconds=1),
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
        seed.create_session(session_id="session-1")
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
    """Start one deterministic run for store tests."""

    return store.start_run(
        run_id="run-1",
        session_id=session_id,
        prompt="hello",
        model="gpt-5.4",
        provider="test",
        started_at=STARTED_AT,
    )

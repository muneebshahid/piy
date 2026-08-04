"""Concurrency tests for the unified SQLite Store."""

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tests.support.store import seed_database, start_run, terminal_outcome
from tile import (
    Aborted,
    ActiveRunError,
    Completed,
    RunAlreadyEndedError,
    RunRecord,
    StorePersistenceError,
)
from tile.store import SQLiteStore
from tile.types import UserMessage


def _finish_run(store: SQLiteStore) -> RunRecord:
    """Finish the shared run with one committed history item."""

    return store.finish_run(
        session_id="session-1",
        run_id="run-1",
        outcome=Completed(value="done"),
        history_delta=[UserMessage(content="hello")],
    )


def _assert_no_run_admitted(store: SQLiteStore) -> None:
    """Assert the failed start committed no run."""

    assert store.list_runs("session-1") == ()


def _assert_start_retry_succeeds(store: SQLiteStore) -> None:
    """Assert admission succeeds once the reader releases its lock."""

    assert start_run(store).run.status == "active"


def _assert_run_still_active(store: SQLiteStore) -> None:
    """Assert the failed finish left the run active with no history."""

    assert store.get_run(session_id="session-1", run_id="run-1").status == "active"
    assert store.get_history("session-1") == ()


def _assert_finish_retry_succeeds(store: SQLiteStore) -> None:
    """Assert finalization succeeds once the reader releases its lock."""

    assert _finish_run(store).status == "completed"
    assert len(store.get_history("session-1")) == 1


def test_concurrent_starts_leave_exactly_one_active_run(tmp_path: Path) -> None:
    """Serialize competing starts across independent Store instances."""

    database_path = tmp_path / "starts.db"
    seed_database(database_path)
    barrier = Barrier(2)

    def start(run_id: str) -> str:
        """Race one run start and report its domain result."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            start_run(store, run_id=run_id, prompt=run_id)
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
        assert sum(run.status == "active" for run in runs) == 1
    finally:
        reopened.close()


def test_concurrent_durable_aborts_are_idempotent(tmp_path: Path) -> None:
    """Allow exactly one escape-hatch caller to transition the active run."""

    database_path = tmp_path / "aborts.db"
    seed_database(database_path, active_run=True)
    barrier = Barrier(2)

    def abort(_: int) -> str:
        """Race one durable abort through an independent Store instance."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            record = store.abort_active_run("session-1")
            return "aborted" if record is not None else "idle"
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(abort, (1, 2)))

    reopened = SQLiteStore(database_path)
    try:
        runs = reopened.list_runs("session-1")
        assert sorted(results) == ["aborted", "idle"]
        assert len(runs) == 1
        assert terminal_outcome(runs[0]) == Aborted(reason="cancelled")
    finally:
        reopened.close()


def test_finish_and_durable_abort_race_preserves_one_valid_winner(
    tmp_path: Path,
) -> None:
    """Keep completion or durable abort intact according to transaction order."""

    database_path = tmp_path / "finish-abort.db"
    seed_database(database_path, active_run=True)
    barrier = Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish_future = executor.submit(_race_finish, database_path, barrier)
        abort_future = executor.submit(_race_abort, database_path, barrier)
        finish_result = finish_future.result()
        abort_result = abort_future.result()

    _assert_finish_abort_winner(database_path, finish_result, abort_result)


@pytest.mark.parametrize(
    ("active_run", "attempt", "assert_unchanged", "assert_recovered"),
    [
        pytest.param(
            False,
            start_run,
            _assert_no_run_admitted,
            _assert_start_retry_succeeds,
            id="start",
        ),
        pytest.param(
            True,
            _finish_run,
            _assert_run_still_active,
            _assert_finish_retry_succeeds,
            id="finish",
        ),
    ],
)
def test_failed_commit_rolls_back_and_store_recovers(
    tmp_path: Path,
    *,
    active_run: bool,
    attempt: Callable[[SQLiteStore], object],
    assert_unchanged: Callable[[SQLiteStore], None],
    assert_recovered: Callable[[SQLiteStore], None],
) -> None:
    """Recover cleanly after a reader prevents a write from committing."""

    database_path = tmp_path / "failed-commit.db"
    seed_database(database_path, active_run=active_run)
    store = _contention_store(database_path)
    reader = _hold_read_transaction(database_path)

    try:
        with pytest.raises(StorePersistenceError, match="database is locked"):
            attempt(store)
        assert_unchanged(store)
        reader.rollback()
        assert_recovered(store)
    finally:
        reader.close()
        store.close()


def _race_finish(database_path: Path, barrier: Barrier) -> str:
    """Race terminal completion against a durable abort."""

    store = SQLiteStore(database_path)
    try:
        barrier.wait()
        _finish_run(store)
        return "finished"
    except RunAlreadyEndedError:
        return "already-ended"
    finally:
        store.close()


def _race_abort(database_path: Path, barrier: Barrier) -> str:
    """Race a durable abort against terminal completion."""

    store = SQLiteStore(database_path)
    try:
        barrier.wait()
        record = store.abort_active_run("session-1")
        return "aborted" if record is not None else "idle"
    finally:
        store.close()


def _assert_finish_abort_winner(
    database_path: Path,
    finish_result: str,
    abort_result: str,
) -> None:
    """Assert the durable terminal state matches the transaction winner."""

    reopened = SQLiteStore(database_path)
    try:
        run = reopened.get_run(session_id="session-1", run_id="run-1")
        if finish_result == "finished":
            assert abort_result == "idle"
            assert run.status == "completed"
            assert len(reopened.get_history("session-1")) == 1
        else:
            assert abort_result == "aborted"
            assert terminal_outcome(run) == Aborted(reason="cancelled")
            assert reopened.get_history("session-1") == ()
    finally:
        reopened.close()


def _contention_store(database_path: Path) -> SQLiteStore:
    """Open a Store that reports lock contention without a long wait."""

    store = SQLiteStore(database_path)
    store._connection.execute("PRAGMA busy_timeout = 1")
    return store


def _hold_read_transaction(database_path: Path) -> sqlite3.Connection:
    """Hold a shared database lock through an explicit read transaction."""

    reader = sqlite3.connect(database_path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT id FROM sessions").fetchall()
    return reader

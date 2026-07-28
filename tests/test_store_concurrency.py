"""Concurrency tests for the unified SQLite Store."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from tile import Aborted, ActiveRunError, Completed, StaleRunError
from tile.store import SQLiteStore
from tile.types import UserMessage
from tests.support.store import create_session, run_record, start_run


def test_concurrent_starts_leave_exactly_one_running_run(tmp_path: Path) -> None:
    """Serialize competing starts across independent Store instances."""

    database_path = tmp_path / "starts.db"
    seed = SQLiteStore(database_path)
    create_session(seed, session_id="session-1")
    seed.close()
    barrier = Barrier(2)

    def start(run_id: str) -> str:
        """Race one run start and report its domain result."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                record=run_record(
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
    create_session(seed, session_id="session-1")
    start_run(seed)
    seed.close()
    barrier = Barrier(2)

    def replace(run_id: str) -> str:
        """Race one replacement through an independent Store instance."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.start_run(
                record=run_record(
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
            barrier.wait()
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
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
                record=run_record(
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
        create_session(seed, session_id="session-1")
        start_run(seed)
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

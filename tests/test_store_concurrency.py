"""Concurrency tests for the unified SQLite Store."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from tile import Aborted, ActiveRunError, Completed, RunAlreadyEndedError
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


def test_concurrent_durable_aborts_are_idempotent(tmp_path: Path) -> None:
    """Allow exactly one escape-hatch caller to transition the active run."""

    database_path = tmp_path / "aborts.db"
    seed = SQLiteStore(database_path)
    create_session(seed, session_id="session-1")
    start_run(seed)
    seed.close()
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
        assert runs[0].outcome == Aborted(reason="cancelled")
    finally:
        reopened.close()


def test_finish_and_durable_abort_race_preserves_one_valid_winner(
    tmp_path: Path,
) -> None:
    """Keep completion or durable abort intact according to transaction order."""

    database_path = tmp_path / "finish-abort.db"
    _seed_running_run(database_path)
    barrier = Barrier(2)

    def finish() -> str:
        """Race terminal completion against the durable abort."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            store.finish_run(
                run_id="run-1",
                outcome=Completed(value="done"),
                history_delta=[UserMessage(content="hello")],
            )
            return "finished"
        except RunAlreadyEndedError:
            return "already-ended"
        finally:
            store.close()

    def abort() -> str:
        """Race the durable abort against terminal completion."""

        store = SQLiteStore(database_path)
        try:
            barrier.wait()
            record = store.abort_active_run("session-1")
            return "aborted" if record is not None else "idle"
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish_future = executor.submit(finish)
        abort_future = executor.submit(abort)
        finish_result = finish_future.result()
        abort_result = abort_future.result()

    reopened = SQLiteStore(database_path)
    try:
        run = reopened.get_run("run-1")
        if finish_result == "finished":
            assert abort_result == "idle"
            assert run.status == "completed"
            assert len(reopened.get_history("session-1")) == 1
        else:
            assert abort_result == "aborted"
            assert run.outcome == Aborted(reason="cancelled")
            assert reopened.get_history("session-1") == ()
    finally:
        reopened.close()


def _seed_running_run(database_path: Path) -> None:
    """Create the shared run used by a finish-versus-abort race."""

    seed = SQLiteStore(database_path)
    try:
        create_session(seed, session_id="session-1")
        start_run(seed)
    finally:
        seed.close()

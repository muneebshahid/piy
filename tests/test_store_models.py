"""Tests for the persistence-first domain models and Store boundary."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.support.store import (
    STARTED_AT,
    corrupt_column,
    create_session,
    persist_outcome,
    start_run,
)
from tile import (
    Aborted,
    ActiveRun,
    AgentFailure,
    Completed,
    ExecutionFailure,
    Failed,
    HistoryItem,
    InvalidHistoryError,
    SessionRecord,
    StorePersistenceError,
    TerminalRun,
)
from tile.store import SQLiteStore, TerminalRunStatus
from tile.types import ConversationItem, UserMessage


def _session_record() -> SessionRecord:
    """Build one deterministic session record."""

    return SessionRecord(
        id="session-1",
        created_at=STARTED_AT,
        updated_at=STARTED_AT,
    )


def _active_record() -> ActiveRun:
    """Build one deterministic active record."""

    return ActiveRun(
        id="run-1",
        session_id="session-1",
        prompt="hello",
        started_at=STARTED_AT,
        model="gpt-5.4",
        provider="openai",
    )


def _terminal_record(
    *,
    outcome: Completed | Failed | Aborted,
) -> TerminalRun:
    """Build one deterministic terminal record snapshot."""

    return TerminalRun(
        id="run-1",
        session_id="session-1",
        prompt="hello",
        started_at=STARTED_AT,
        ended_at=STARTED_AT + timedelta(seconds=1),
        model="gpt-5.4",
        provider="openai",
        outcome=outcome,
    )


def _history_item(
    *,
    item: ConversationItem | None = None,
) -> HistoryItem:
    """Build one deterministic committed history item."""

    return HistoryItem(
        id="history-1",
        session_id="session-1",
        run_id="run-1",
        position=0,
        item=item if item is not None else UserMessage(content="hello"),
        created_at=STARTED_AT,
    )


def test_persistent_records_are_frozen() -> None:
    """Prevent callers from replacing fields on authoritative records."""

    session = _session_record()
    run = _active_record()
    history_item = _history_item()

    with pytest.raises(ValidationError):
        session.updated_at = STARTED_AT + timedelta(seconds=1)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        run.prompt = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        history_item.position = 2  # ty: ignore[invalid-assignment]


def test_session_record_validates_its_lifecycle_timestamps() -> None:
    """Require stable aware timestamps in chronological order."""

    with pytest.raises(ValidationError, match="timezone-aware"):
        SessionRecord(
            id="session-1",
            created_at=datetime(2026, 7, 26, 12, 0),  # noqa: DTZ001
            updated_at=STARTED_AT,
        )

    with pytest.raises(ValidationError, match="updated before"):
        SessionRecord(
            id="session-1",
            created_at=STARTED_AT,
            updated_at=STARTED_AT - timedelta(seconds=1),
        )


def test_conversation_item_is_discriminated_by_role() -> None:
    """Reject payloads whose role is outside the typed conversation contract."""

    adapter = TypeAdapter(ConversationItem)

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"role": "unknown", "content": "hello"})


def test_history_item_rejects_a_negative_position() -> None:
    """Reject envelopes placed before the start of committed history."""

    with pytest.raises(ValidationError, match="negative"):
        HistoryItem(
            id="history-1",
            session_id="session-1",
            run_id="run-1",
            position=-1,
            item=UserMessage(content="hello"),
            created_at=STARTED_AT,
        )


def test_get_history_rejects_a_persisted_role_contradicting_its_payload(
    tmp_path: Path,
) -> None:
    """Surface stored role and payload disagreement as InvalidHistoryError."""

    database_path = tmp_path / "contradicting-role.db"
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
    corrupt_column(
        database_path, table="history_items", column="role", value="assistant"
    )

    reopened = SQLiteStore(database_path)
    try:
        with pytest.raises(InvalidHistoryError, match="contradicts"):
            reopened.get_history("session-1")
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        pytest.param(Completed(value="done"), "completed", id="completed"),
        pytest.param(
            Failed(cause=AgentFailure(reason="cannot deliver")),
            "failed",
            id="agent-failure",
        ),
        pytest.param(
            Failed(
                cause=ExecutionFailure(
                    origin="execution",
                    exception_type="RuntimeError",
                    message="runtime failed",
                )
            ),
            "failed",
            id="execution-failure",
        ),
        pytest.param(
            Aborted(reason="cancelled"),
            "aborted",
            id="cancelled",
        ),
    ],
)
def test_terminal_run_derives_status_from_its_outcome(
    outcome: Completed | Failed | Aborted,
    expected_status: TerminalRunStatus,
) -> None:
    """Derive the terminal status from the serializable outcome."""

    finished = _terminal_record(outcome=outcome)

    assert finished.status == expected_status
    assert finished.outcome == outcome
    assert finished.prompt == "hello"
    assert finished.model == "gpt-5.4"
    assert finished.provider == "openai"
    assert finished.model_dump()["status"] == expected_status
    assert TerminalRun.model_validate(finished.model_dump()) == finished


def test_active_run_cannot_represent_terminal_data() -> None:
    """Drop terminal payload keys because an active record has no such fields."""

    hydrated = ActiveRun.model_validate(
        _active_record().model_dump() | {"ended_at": STARTED_AT + timedelta(seconds=1)}
    )

    assert hydrated == _active_record()


def test_terminal_run_requires_an_end_timestamp() -> None:
    """Reject a terminal record constructed without its end timestamp."""

    with pytest.raises(ValidationError, match="ended_at"):
        TerminalRun.model_validate(
            _active_record().model_dump() | {"outcome": Completed(value="done")}
        )


def test_terminal_run_rejects_ending_before_start() -> None:
    """Reject a terminal record whose end precedes its start."""

    with pytest.raises(ValidationError, match="end before"):
        TerminalRun.model_validate(
            _terminal_record(outcome=Completed(value="done")).model_dump()
            | {"ended_at": STARTED_AT - timedelta(seconds=1)}
        )


def test_get_run_rejects_a_stored_status_contradicting_its_outcome(
    tmp_path: Path,
) -> None:
    """Surface stored status and outcome disagreement as a persistence error."""

    database_path = tmp_path / "contradicting-status.db"
    store = SQLiteStore(database_path)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[],
        )
    finally:
        store.close()
    corrupt_column(database_path, table="runs", column="status", value="aborted")

    reopened = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError, match="contradicts"):
            reopened.get_run("session-1", "run-1")
    finally:
        reopened.close()


def test_get_run_rejects_a_terminal_row_missing_its_terminal_data(
    tmp_path: Path,
) -> None:
    """Surface a terminal row stripped of its outcome as a persistence error."""

    database_path = tmp_path / "missing-terminal-data.db"
    store = SQLiteStore(database_path)
    try:
        create_session(store, session_id="session-1")
        start_run(store)
        persist_outcome(
            store,
            outcome=Completed(value="done"),
            history_delta=[],
        )
    finally:
        store.close()
    corrupt_column(database_path, table="runs", column="outcome_json", value=None)

    reopened = SQLiteStore(database_path)
    try:
        with pytest.raises(StorePersistenceError, match="missing terminal data"):
            reopened.get_run("session-1", "run-1")
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param(
            AgentFailure(reason="cannot deliver"),
            id="agent-failure",
        ),
        pytest.param(
            ExecutionFailure(
                origin="execution",
                exception_type="RuntimeError",
                message="runtime failed",
            ),
            id="execution-failure",
        ),
    ],
)
def test_failure_causes_round_trip_without_losing_their_kind(
    cause: AgentFailure | ExecutionFailure,
) -> None:
    """Preserve every failure cause kind through serialization."""

    outcome = Failed(cause=cause)
    adapter = TypeAdapter(Failed)

    loaded = adapter.validate_json(outcome.model_dump_json())

    assert type(loaded.cause) is type(cause)
    assert loaded == outcome

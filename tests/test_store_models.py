"""Tests for the persistence-first domain models and Store boundary."""

from datetime import datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from tile import (
    Aborted,
    AgentFailure,
    Completed,
    ExecutionFailure,
    Failed,
    HistoryItem,
    RunRecord,
    SessionRecord,
)
from tile.store import TerminalRunStatus
from tile.types import ConversationItem, UserMessage
from tests.support.store import STARTED_AT


def _session_record() -> SessionRecord:
    """Build one deterministic session record."""

    return SessionRecord(
        id="session-1",
        created_at=STARTED_AT,
        updated_at=STARTED_AT,
    )


def _running_record() -> RunRecord:
    """Build one deterministic running record."""

    return RunRecord(
        id="run-1",
        session_id="session-1",
        prompt="hello",
        status="running",
        started_at=STARTED_AT,
        model="gpt-5.4",
        provider="openai",
    )


def _terminal_record(
    *,
    outcome: Completed | Failed | Aborted,
    status: TerminalRunStatus,
) -> RunRecord:
    """Build one deterministic terminal record snapshot."""

    return RunRecord(
        id="run-1",
        session_id="session-1",
        prompt="hello",
        status=status,
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
    run = _running_record()
    history_item = _history_item()

    with pytest.raises(ValidationError):
        session.updated_at = STARTED_AT + timedelta(seconds=1)
    with pytest.raises(ValidationError):
        run.status = "completed"
    with pytest.raises(ValidationError):
        history_item.position = 2


def test_session_record_validates_its_lifecycle_timestamps() -> None:
    """Require stable aware timestamps in chronological order."""

    with pytest.raises(ValidationError, match="timezone-aware"):
        SessionRecord(
            id="session-1",
            created_at=datetime(2026, 7, 26, 12, 0),
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
def test_run_record_accepts_every_consistent_terminal_outcome(
    outcome: Completed | Failed | Aborted,
    expected_status: TerminalRunStatus,
) -> None:
    """Keep terminal status consistent with its serializable outcome."""

    finished = _terminal_record(outcome=outcome, status=expected_status)

    assert finished.status == expected_status
    assert finished.outcome == outcome
    assert finished.prompt == "hello"
    assert finished.model == "gpt-5.4"
    assert finished.provider == "openai"


@pytest.mark.parametrize(
    ("values", "match"),
    [
        pytest.param(
            _running_record().model_dump()
            | {"status": "completed", "outcome": Completed(value="done")},
            "end timestamp",
            id="terminal-status-without-end",
        ),
        pytest.param(
            _terminal_record(
                outcome=Completed(value="done"),
                status="completed",
            ).model_dump()
            | {"status": "aborted"},
            "contradicts",
            id="status-conflicts-with-outcome",
        ),
    ],
)
def test_terminal_run_rejects_inconsistent_lifecycle_snapshots(
    values: dict[str, object],
    match: str,
) -> None:
    """Reject terminal snapshots whose redundant lifecycle facts disagree."""

    with pytest.raises(ValidationError, match=match):
        RunRecord.model_validate(values)


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

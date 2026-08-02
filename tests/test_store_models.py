"""Tests for the persistence-first domain models and Store boundary."""

from datetime import UTC, datetime, timedelta
from typing import get_type_hints

import pytest
from pydantic import TypeAdapter, ValidationError

from tile import (
    Aborted,
    AgentFailure,
    Completed,
    ExecutionFailure,
    Failed,
    HistoryItem,
    RunHandle,
    RunRecord,
    SessionRecord,
    Store,
)
from tile.store import TerminalRunStatus
from tile.types import AssistantTurn, ConversationItem, UserMessage

CREATED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def test_persistent_records_are_frozen() -> None:
    """Prevent callers from replacing fields on authoritative records."""

    session = _session_record()
    run = _running_record()
    history_item = _history_item()

    with pytest.raises(ValidationError):
        session.updated_at = CREATED_AT + timedelta(seconds=1)
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
            updated_at=CREATED_AT,
        )

    with pytest.raises(ValidationError, match="updated before"):
        SessionRecord(
            id="session-1",
            created_at=CREATED_AT,
            updated_at=CREATED_AT - timedelta(seconds=1),
        )


def test_persistent_records_are_passive_store_outputs() -> None:
    """Keep caller intent and lifecycle transitions out of record models."""

    assert "name" not in SessionRecord.model_fields
    assert not hasattr(SessionRecord, "create")
    assert not hasattr(RunRecord, "start")
    assert not hasattr(RunRecord, "finish")


def test_history_item_round_trips_typed_conversation_payloads() -> None:
    """Deserialize adapter-shaped JSON back into the conversation union."""

    adapter = TypeAdapter(HistoryItem)
    user_item = _history_item(item=UserMessage(content="hello"))
    assistant_item = _history_item(
        item=AssistantTurn(response_id="response-1"),
    )

    loaded_user = adapter.validate_json(user_item.model_dump_json())
    loaded_assistant = adapter.validate_json(assistant_item.model_dump_json())

    assert isinstance(loaded_user.item, UserMessage)
    assert isinstance(loaded_assistant.item, AssistantTurn)
    assert loaded_user == user_item
    assert loaded_assistant == assistant_item


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


def test_run_record_requires_a_provider_at_creation() -> None:
    """Require execution identity before a run can enter persistence."""

    values = _running_record().model_dump(exclude={"provider"})

    with pytest.raises(ValidationError, match="provider"):
        RunRecord.model_validate(values)


def test_terminal_run_requires_an_end_timestamp() -> None:
    """Reject a persisted terminal snapshot without complete lifecycle data."""

    values = _running_record().model_dump()
    values.update(status="completed", outcome=Completed(value="done"))

    with pytest.raises(ValidationError, match="end timestamp"):
        RunRecord.model_validate(values)


def test_terminal_run_rejects_a_status_that_conflicts_with_its_outcome() -> None:
    """Reject terminal snapshots whose redundant lifecycle facts disagree."""

    values = _terminal_record(
        outcome=Completed(value="done"),
        status="completed",
    ).model_dump()
    values.update(status="aborted")

    with pytest.raises(ValidationError, match="contradicts"):
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


def test_store_contract_exposes_only_lifecycle_operations() -> None:
    """Keep unrestricted history and run mutations out of the Store API."""

    methods = {
        name
        for name, value in Store.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert methods == {
        "abort_active_run",
        "create_session",
        "finish_run",
        "fork_session",
        "get_history",
        "get_run",
        "get_session",
        "list_runs",
        "list_sessions",
        "start_run",
    }
    assert "append_history" not in methods
    assert "update_run" not in methods
    assert get_type_hints(Store.get_history)["return"]


def test_live_run_type_is_named_run_handle() -> None:
    """Distinguish the live execution handle from a persistent RunRecord."""

    assert RunHandle.__name__ == "RunHandle"
    assert RunHandle is not RunRecord


def _session_record() -> SessionRecord:
    """Build one deterministic session record."""

    return SessionRecord(
        id="session-1",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _running_record() -> RunRecord:
    """Build one deterministic running record."""

    return RunRecord(
        id="run-1",
        session_id="session-1",
        prompt="hello",
        status="running",
        started_at=CREATED_AT,
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
        started_at=CREATED_AT,
        ended_at=CREATED_AT + timedelta(seconds=1),
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
        created_at=CREATED_AT,
    )

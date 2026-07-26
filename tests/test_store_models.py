"""Tests for the persistence-first domain models and Store boundary."""

from datetime import UTC, datetime, timedelta
from typing import get_type_hints

import pytest
from pydantic import TypeAdapter, ValidationError

from tile import (
    Aborted,
    AgentFailure,
    Completed,
    Failed,
    HistoryItem,
    PersistenceFailure,
    RunHandle,
    RunRecord,
    SessionRecord,
    Store,
)
from tile.types import AssistantTurn, ConversationItem, UserMessage

CREATED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def test_persistent_records_are_frozen() -> None:
    """Prevent callers from replacing fields on authoritative records."""

    session = _session_record()
    run = _running_record()
    history_item = _history_item()

    with pytest.raises(ValidationError):
        session.name = "Changed"
    with pytest.raises(ValidationError):
        run.status = "completed"
    with pytest.raises(ValidationError):
        history_item.position = 2


def test_session_record_validates_its_lifecycle_timestamps() -> None:
    """Require stable aware timestamps in chronological order."""

    with pytest.raises(ValidationError, match="timezone-aware"):
        SessionRecord(
            session_id="session-1",
            created_at=datetime(2026, 7, 26, 12, 0),
            updated_at=CREATED_AT,
        )

    with pytest.raises(ValidationError, match="updated before"):
        SessionRecord(
            session_id="session-1",
            created_at=CREATED_AT,
            updated_at=CREATED_AT - timedelta(seconds=1),
        )


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
                cause=PersistenceFailure(
                    operation="finish_run",
                    exception_type="OSError",
                    message="disk full",
                )
            ),
            "failed",
            id="persistence-failure",
        ),
        pytest.param(
            Aborted(reason="cancelled"),
            "aborted",
            id="cancelled",
        ),
        pytest.param(
            Aborted(reason="replaced"),
            "aborted",
            id="replaced",
        ),
    ],
)
def test_run_record_derives_status_from_every_outcome(
    outcome: Completed | Failed | Aborted,
    expected_status: str,
) -> None:
    """Keep terminal status consistent with its serializable outcome."""

    finished = _running_record().finish(outcome=outcome, ended_at=CREATED_AT)

    assert finished.status == expected_status
    assert finished.outcome == outcome
    assert finished.prompt == "hello"


def test_run_record_finish_rejects_a_naive_end_timestamp() -> None:
    """Reject an end timestamp whose timezone would make ordering ambiguous."""

    with pytest.raises(ValueError, match="timezone-aware"):
        _running_record().finish(
            outcome=Completed(value="done"),
            ended_at=datetime(2026, 7, 26, 12, 1),
        )


def test_failure_causes_round_trip_without_losing_their_kind() -> None:
    """Preserve agent, execution, and persistence failure distinctions."""

    outcome = Failed(
        cause=PersistenceFailure(
            operation="finish_run",
            exception_type="OSError",
            message="disk full",
        )
    )
    adapter = TypeAdapter(Failed)

    loaded = adapter.validate_json(outcome.model_dump_json())

    assert isinstance(loaded.cause, PersistenceFailure)
    assert loaded == outcome


def test_store_contract_exposes_only_lifecycle_operations() -> None:
    """Keep unrestricted history and run mutations out of the Store API."""

    methods = {
        name
        for name, value in Store.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert methods == {
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
        session_id="session-1",
        name="Session",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _running_record() -> RunRecord:
    """Build one deterministic running record."""

    return RunRecord(
        run_id="run-1",
        session_id="session-1",
        prompt="hello",
        status="running",
        started_at=CREATED_AT,
        model="gpt-5.4",
        provider="openai",
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

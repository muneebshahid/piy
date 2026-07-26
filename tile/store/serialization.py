"""Typed mapping between persistent domain records and SQLite values."""

from datetime import datetime
from typing import TypeAlias, cast
from uuid import uuid4

from pydantic import TypeAdapter

from tile.result import RunOutcome
from tile.store.base import InvalidHistoryError
from tile.store.models import HistoryItem, RunRecord, RunStatus, SessionRecord
from tile.types.conversation import ConversationItem

SessionRow: TypeAlias = tuple[str, str | None, str, str]
RunRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    str | None,
    str,
    str | None,
    str | None,
]
HistoryRow: TypeAlias = tuple[str, str, str, int, str, str, str]
HistoryInsertRow: TypeAlias = tuple[str, str, str, int, str, str, str]
TerminalRunValues: TypeAlias = tuple[str, str, str, str | None, str, str]

_OUTCOME_ADAPTER = TypeAdapter(RunOutcome)
_CONVERSATION_ITEM_ADAPTER = TypeAdapter(ConversationItem)


def session_values(record: SessionRecord) -> SessionRow:
    """Serialize a session record into SQLite column values."""

    return (
        record.session_id,
        record.name,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
    )


def session_from_row(row: SessionRow) -> SessionRecord:
    """Deserialize one SQLite session row."""

    session_id, name, created_at, updated_at = row
    return SessionRecord(
        session_id=session_id,
        name=name,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
    )


def run_values(record: RunRecord) -> RunRow:
    """Serialize a run record into SQLite column values."""

    return (
        record.run_id,
        record.session_id,
        record.prompt,
        record.status,
        record.started_at.isoformat(),
        record.ended_at.isoformat() if record.ended_at is not None else None,
        record.model,
        record.provider,
        _dump_outcome(record.outcome),
    )


def terminal_run_values(record: RunRecord) -> TerminalRunValues:
    """Serialize fields used by the conditional terminal update."""

    if record.ended_at is None or record.outcome is None:
        raise ValueError("A terminal update requires terminal run data.")
    outcome_json = _dump_outcome(record.outcome)
    if outcome_json is None:
        raise ValueError("A terminal update requires a serialized outcome.")
    return (
        record.status,
        record.ended_at.isoformat(),
        record.model,
        record.provider,
        outcome_json,
        record.run_id,
    )


def run_from_row(row: RunRow) -> RunRecord:
    """Deserialize one SQLite run row."""

    (
        run_id,
        session_id,
        prompt,
        status,
        started_at,
        ended_at,
        model,
        provider,
        outcome_json,
    ) = row
    return RunRecord(
        run_id=run_id,
        session_id=session_id,
        prompt=prompt,
        status=cast("RunStatus", status),
        started_at=datetime.fromisoformat(started_at),
        ended_at=datetime.fromisoformat(ended_at) if ended_at is not None else None,
        model=model,
        provider=provider,
        outcome=_load_outcome(outcome_json),
    )


def history_values(
    *,
    item: ConversationItem,
    run_id: str,
    session_id: str,
    position: int,
    created_at: datetime,
) -> HistoryInsertRow:
    """Serialize one new committed history envelope."""

    return (
        str(uuid4()),
        session_id,
        run_id,
        position,
        item.role,
        dump_conversation_item(item),
        created_at.isoformat(),
    )


def history_from_row(row: HistoryRow) -> HistoryItem:
    """Deserialize one SQLite history row into a typed envelope."""

    item_id, session_id, run_id, position, role, payload_json, created_at = row
    item = _CONVERSATION_ITEM_ADAPTER.validate_json(payload_json)
    if item.role != role:
        raise InvalidHistoryError(
            f"Stored history role {role!r} contradicts payload role {item.role!r}."
        )
    return HistoryItem(
        id=item_id,
        session_id=session_id,
        run_id=run_id,
        position=position,
        item=item,
        created_at=datetime.fromisoformat(created_at),
    )


def dump_conversation_item(item: ConversationItem) -> str:
    """Serialize one typed conversation item."""

    return _CONVERSATION_ITEM_ADAPTER.dump_json(item).decode()


def _dump_outcome(outcome: RunOutcome | None) -> str | None:
    """Serialize one typed run outcome."""

    if outcome is None:
        return None
    return _OUTCOME_ADAPTER.dump_json(outcome).decode()


def _load_outcome(payload_json: str | None) -> RunOutcome | None:
    """Deserialize one typed run outcome."""

    if payload_json is None:
        return None
    return _OUTCOME_ADAPTER.validate_json(payload_json)

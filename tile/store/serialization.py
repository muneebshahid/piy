"""Typed mapping between persistent domain records and SQLite values."""

from datetime import datetime
from uuid import uuid4

from pydantic import TypeAdapter

from tile.result import RunOutcome
from tile.store.base import InvalidHistoryError
from tile.store.models import (
    ActiveRun,
    HistoryItem,
    RunRecord,
    RunStatus,
    SessionRecord,
    TerminalRun,
)
from tile.types.conversation import ConversationItem

type SessionRow = tuple[str, str, str]
type RunRow = tuple[
    str,
    str,
    str,
    RunStatus,
    str,
    str | None,
    str,
    str,
    str | None,
]
type HistoryRow = tuple[str, str, str, int, str, str, str]
type TerminalRunValues = tuple[str, str, str, str, str]

_OUTCOME_ADAPTER = TypeAdapter(RunOutcome)
_CONVERSATION_ITEM_ADAPTER = TypeAdapter(ConversationItem)


def session_values(record: SessionRecord) -> SessionRow:
    """Serialize a session record into SQLite column values."""

    return (
        record.id,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
    )


def session_from_row(row: SessionRow) -> SessionRecord:
    """Deserialize one SQLite session row."""

    session_id, created_at, updated_at = row
    return SessionRecord(
        id=session_id,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
    )


def run_values(record: ActiveRun) -> RunRow:
    """Serialize a newly accepted active run into SQLite column values."""

    return (
        record.id,
        record.session_id,
        record.prompt,
        record.status,
        record.started_at.isoformat(),
        None,
        record.model,
        record.provider,
        None,
    )


def terminal_run_values(record: TerminalRun) -> TerminalRunValues:
    """Serialize fields used by the conditional terminal update."""

    return (
        record.status,
        record.ended_at.isoformat(),
        _dump_outcome(record.outcome),
        record.id,
        record.session_id,
    )


def run_from_row(row: RunRow) -> RunRecord:
    """Deserialize one SQLite run row, rehydrating its lifecycle state."""

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
    if status == "active":
        return active_run_from_row(row)
    if ended_at is None or outcome_json is None:
        raise ValueError(f"Terminal run row {run_id!r} is missing terminal data.")
    record = TerminalRun(
        id=run_id,
        session_id=session_id,
        prompt=prompt,
        started_at=datetime.fromisoformat(started_at),
        ended_at=datetime.fromisoformat(ended_at),
        model=model,
        provider=provider,
        outcome=_load_outcome(outcome_json),
    )
    if record.status != status:
        raise ValueError(
            f"Stored run status {status!r} contradicts the stored outcome, "
            f"which implies {record.status!r}."
        )
    return record


def active_run_from_row(row: RunRow) -> ActiveRun:
    """Deserialize one SQLite run row that must hold an active run."""

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
    if status != "active":
        raise ValueError(f"Run row {run_id!r} is not active.")
    if ended_at is not None or outcome_json is not None:
        raise ValueError(f"Active run row {run_id!r} carries terminal data.")
    return ActiveRun(
        id=run_id,
        session_id=session_id,
        prompt=prompt,
        started_at=datetime.fromisoformat(started_at),
        model=model,
        provider=provider,
    )


def history_values(
    *,
    item: ConversationItem,
    run_id: str,
    session_id: str,
    position: int,
    created_at: datetime,
) -> HistoryRow:
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

    history_item_id, session_id, run_id, position, role, payload_json, created_at = row
    item = _CONVERSATION_ITEM_ADAPTER.validate_json(payload_json)
    if item.role != role:
        raise InvalidHistoryError(
            f"Stored history role {role!r} contradicts payload role {item.role!r}."
        )
    return HistoryItem(
        id=history_item_id,
        session_id=session_id,
        run_id=run_id,
        position=position,
        item=item,
        created_at=datetime.fromisoformat(created_at),
    )


def dump_conversation_item(item: ConversationItem) -> str:
    """Serialize one typed conversation item."""

    return _CONVERSATION_ITEM_ADAPTER.dump_json(item).decode()


def _dump_outcome(outcome: RunOutcome) -> str:
    """Serialize one typed run outcome."""

    return _OUTCOME_ADAPTER.dump_json(outcome).decode()


def _load_outcome(payload_json: str) -> RunOutcome:
    """Deserialize one typed run outcome."""

    return _OUTCOME_ADAPTER.validate_json(payload_json)

"""Persistent domain records shared by store contracts and adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self, assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)

from tile.result import Aborted, Completed, Failed, RunOutcome
from tile.types.conversation import ConversationItem

type RunStatus = Literal["active", "completed", "failed", "aborted"]
type TerminalRunStatus = Literal["completed", "failed", "aborted"]


class SessionRecord(BaseModel):
    """Immutable metadata for one persistent conversation session."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        return _normalize_timestamp(value, label="Session")

    @model_validator(mode="after")
    def _validate_timestamps(self) -> Self:
        """Reject a session update timestamp before its creation."""

        if self.updated_at < self.created_at:
            raise ValueError("A session cannot be updated before it is created.")
        return self


class _RunBase(BaseModel):
    """Shared identity and admission state for one submitted prompt run."""

    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    prompt: str
    started_at: datetime
    model: str
    provider: str

    @field_validator("started_at")
    @classmethod
    def _normalize_started_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        return _normalize_timestamp(value, label="Run")


class ActiveRun(_RunBase):
    """Immutable persistent state for a run that has not ended."""

    status: Literal["active"] = "active"


class TerminalRun(_RunBase):
    """Immutable persistent state for a run that has ended.

    ``status`` is derived from ``outcome``: dumps include the derived value,
    inbound ``status`` keys are ignored, and the stored DB column is
    validated against the derivation on rehydration.
    """

    ended_at: datetime
    outcome: RunOutcome

    @computed_field
    @property
    def status(self) -> TerminalRunStatus:
        """Return the terminal status implied by the outcome."""

        return terminal_status_for(self.outcome)

    @field_validator("ended_at")
    @classmethod
    def _normalize_ended_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        return _normalize_timestamp(value, label="Run")

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> Self:
        """Reject a run that ends before it starts."""

        if self.ended_at < self.started_at:
            raise ValueError("A run cannot end before it starts.")
        return self


RunRecord = ActiveRun | TerminalRun
"""One persistent run in either lifecycle state.

A plain union rather than a ``type`` statement so ``isinstance`` checks
against the alias keep working.
"""


class HistoryItem(BaseModel):
    """A frozen envelope containing a defensive conversation-item snapshot."""

    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    run_id: str
    position: int
    item: ConversationItem
    created_at: datetime

    @field_validator("position")
    @classmethod
    def _validate_position(cls, value: int) -> int:
        """Require a non-negative session-local history position."""

        if value < 0:
            raise ValueError("A history position cannot be negative.")
        return value

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        return _normalize_timestamp(value, label="History item")


class StartedRun(BaseModel):
    """Atomic bootstrap state for one newly started run."""

    model_config = ConfigDict(frozen=True)

    run: ActiveRun
    committed_history: tuple[HistoryItem, ...]


def new_session_record(*, session_id: str) -> SessionRecord:
    """Create Store-owned metadata for a newly persisted session."""

    now = datetime.now(UTC)
    return SessionRecord(
        id=session_id,
        created_at=now,
        updated_at=now,
    )


def new_active_run_record(
    *,
    run_id: str,
    session_id: str,
    prompt: str,
    model: str,
    provider: str,
) -> ActiveRun:
    """Create Store-owned persistent state for a newly accepted run."""

    return ActiveRun(
        id=run_id,
        session_id=session_id,
        prompt=prompt,
        started_at=datetime.now(UTC),
        model=model,
        provider=provider,
    )


def terminal_run_record(
    active: ActiveRun,
    *,
    outcome: RunOutcome,
) -> TerminalRun:
    """Create terminal persistent state from an authoritative active record."""

    terminal_time = datetime.now(UTC)
    return TerminalRun(
        id=active.id,
        session_id=active.session_id,
        prompt=active.prompt,
        started_at=active.started_at,
        ended_at=max(terminal_time, active.started_at),
        model=active.model,
        provider=active.provider,
        outcome=outcome.model_copy(deep=True),
    )


def terminal_status_for(outcome: RunOutcome) -> TerminalRunStatus:
    """Return the terminal run status implied by an outcome."""

    match outcome:
        case Completed():
            return "completed"
        case Failed():
            return "failed"
        case Aborted():
            return "aborted"
        case _:
            assert_never(outcome)


def _normalize_timestamp(value: datetime, *, label: str) -> datetime:
    """Require one timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} timestamps must be timezone-aware.")
    return value.astimezone(UTC)

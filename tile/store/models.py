"""Persistent domain records shared by store contracts and adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tile.result import Aborted, Completed, Failed, RunOutcome
from tile.types.conversation import ConversationItem

type RunStatus = Literal["running", "completed", "failed", "aborted"]
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


class RunRecord(BaseModel):
    """Immutable persistent state for one submitted prompt run."""

    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    prompt: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    model: str
    provider: str
    outcome: RunOutcome | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware timestamps and normalize them to UTC."""

        if value is None:
            return None
        return _normalize_timestamp(value, label="Run")

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> Self:
        """Reject fields that contradict the run lifecycle."""

        if self.status == "running":
            if self.ended_at is not None or self.outcome is not None:
                raise ValueError("A running run cannot have terminal data.")
            return self

        self._validate_terminal_lifecycle()
        return self

    def _validate_terminal_lifecycle(self) -> None:
        """Validate timestamps, outcome presence, and derived terminal status."""

        if self.ended_at is None:
            raise ValueError("A terminal run must have an end timestamp.")
        if self.ended_at < self.started_at:
            raise ValueError("A run cannot end before it starts.")
        if self.outcome is None:
            raise ValueError("A terminal run must have an outcome.")
        implied_status = terminal_status_for(self.outcome)
        if self.status != implied_status:
            raise ValueError(
                f"Status {self.status!r} contradicts the terminal outcome, "
                f"which implies {implied_status!r}."
            )


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

    run: RunRecord
    committed_history: tuple[HistoryItem, ...]


def new_session_record(*, session_id: str) -> SessionRecord:
    """Create Store-owned metadata for a newly persisted session."""

    now = datetime.now(UTC)
    return SessionRecord(
        id=session_id,
        created_at=now,
        updated_at=now,
    )


def new_running_run_record(
    *,
    run_id: str,
    session_id: str,
    prompt: str,
    model: str,
    provider: str,
) -> RunRecord:
    """Create Store-owned persistent state for a newly accepted run."""

    return RunRecord(
        id=run_id,
        session_id=session_id,
        prompt=prompt,
        status="running",
        started_at=datetime.now(UTC),
        model=model,
        provider=provider,
    )


def terminal_run_record(
    running: RunRecord,
    *,
    outcome: RunOutcome,
) -> RunRecord:
    """Create terminal persistent state from an authoritative running record."""

    terminal_time = datetime.now(UTC)
    return RunRecord(
        id=running.id,
        session_id=running.session_id,
        prompt=running.prompt,
        status=terminal_status_for(outcome),
        started_at=running.started_at,
        ended_at=max(terminal_time, running.started_at),
        model=running.model,
        provider=running.provider,
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

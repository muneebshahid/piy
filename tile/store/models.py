"""Persistent domain records shared by store contracts and adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tile.result import Aborted, Completed, Failed, RunOutcome
from tile.types.conversation import ConversationItem

RunStatus: TypeAlias = Literal["running", "completed", "failed", "aborted"]
TerminalRunStatus: TypeAlias = Literal["completed", "failed", "aborted"]


class SessionRecord(BaseModel):
    """Immutable metadata for one persistent conversation session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    name: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        name: str | None = None,
    ) -> SessionRecord:
        """Create a new session record at the current time."""

        now = datetime.now(UTC)
        return cls(
            session_id=session_id,
            name=name,
            created_at=now,
            updated_at=now,
        )

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

    run_id: str
    session_id: str
    prompt: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    model: str
    provider: str
    outcome: RunOutcome | None = None

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        session_id: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> RunRecord:
        """Create a new running record at the current time."""

        return cls(
            run_id=run_id,
            session_id=session_id,
            prompt=prompt,
            status="running",
            started_at=datetime.now(UTC),
            model=model,
            provider=provider,
        )

    def finish(
        self,
        *,
        outcome: RunOutcome,
    ) -> RunRecord:
        """Return the terminal form while preserving execution identity.

        Status is derived from the outcome, and the end timestamp is clamped
        to the start so a backward clock step cannot create invalid history.
        """

        if self.status != "running":
            raise ValueError("Only a running run can be finished.")
        terminal_time = datetime.now(UTC)
        return RunRecord(
            run_id=self.run_id,
            session_id=self.session_id,
            prompt=self.prompt,
            status=terminal_status_for(outcome),
            started_at=self.started_at,
            ended_at=max(terminal_time, self.started_at),
            model=self.model,
            provider=self.provider,
            outcome=outcome,
        )

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
    """One immutable item in a session's committed conversation timeline."""

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
    replaced_run_id: str | None = None


def terminal_status_for(outcome: RunOutcome) -> TerminalRunStatus:
    """Return the terminal run status implied by an outcome."""

    if isinstance(outcome, Completed):
        return "completed"
    if isinstance(outcome, Failed):
        return "failed"
    if isinstance(outcome, Aborted):
        return "aborted"
    raise TypeError(f"Unsupported run outcome: {type(outcome).__name__}")


def _normalize_timestamp(value: datetime, *, label: str) -> datetime:
    """Require one timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} timestamps must be timezone-aware.")
    return value.astimezone(UTC)

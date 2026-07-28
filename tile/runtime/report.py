"""Process-local terminal report for one run."""

from __future__ import annotations

from dataclasses import dataclass

from tile.result import RunOutcome
from tile.store.models import RunRecord, RunStatus
from tile.types.conversation import AssistantTurn, ConversationItem


@dataclass(frozen=True)
class RunReport:
    """Collect the durable result and local diagnostics of a finished run.

    ``record`` is the Store-confirmed terminal record when ``persisted`` is
    true. When finalization fails, it is a process-local terminal projection
    that preserves the execution outcome while ``finalization_error`` explains
    why the Store could not confirm it.
    """

    record: RunRecord
    history_delta: tuple[ConversationItem, ...]
    execution_error: BaseException | None = None
    finalization_error: BaseException | None = None

    @property
    def outcome(self) -> RunOutcome:
        """Return the terminal outcome carried by the report record."""

        outcome = self.record.outcome
        if outcome is None:
            raise RuntimeError("A run report requires a terminal record.")
        return outcome

    @property
    def status(self) -> RunStatus:
        """Return the lifecycle status derived by the terminal record."""

        return self.record.status

    @property
    def persisted(self) -> bool:
        """Return whether finalization produced a Store-confirmed record."""

        return self.finalization_error is None

    @property
    def last_assistant_turn(self) -> AssistantTurn | None:
        """Return the latest assistant turn produced during this run."""

        for item in reversed(self.history_delta):
            if isinstance(item, AssistantTurn):
                return item
        return None

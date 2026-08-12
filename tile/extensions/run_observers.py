"""Passive run-event observation contracts and delivery."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tile.events import AgentEvent


@dataclass(frozen=True)
class RunEvent:
    """One run event enriched with its durable run identity."""

    session_id: str
    run_id: str
    event: AgentEvent


type RunObserver = Callable[[RunEvent], None]


class RunObservers:
    """Publish run events to passive, failure-isolated observers."""

    def __init__(self, observers: Sequence[RunObserver] = ()) -> None:
        """Freeze observers in registration order."""

        self._observers = tuple(observers)

    def publish(self, event: RunEvent) -> None:
        """Notify every observer without exposing mutable runtime state."""

        for observer in self._observers:
            try:
                observer(_copy_run_event(event))
            except Exception:
                _LOGGER.exception(
                    "Tile run observer failed for %s/%s event %s.",
                    event.session_id,
                    event.run_id,
                    event.event.type,
                )


_LOGGER = logging.getLogger(__name__)


def _copy_run_event(event: RunEvent) -> RunEvent:
    """Return an observer-owned copy of one published run event."""

    return RunEvent(
        session_id=event.session_id,
        run_id=event.run_id,
        event=event.event.model_copy(deep=True),
    )

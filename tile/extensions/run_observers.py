"""Passive run-event stream contracts and execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

from tile.events import AgentEvent


@dataclass(frozen=True)
class RunEventStream:
    """One run's identity and independently consumable event iterator."""

    session_id: str
    run_id: str
    events: AsyncIterator[AgentEvent]

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        """Iterate over this observer's ordered run events."""

        return self.events


type RunObserver = Callable[[RunEventStream], Awaitable[None]]
type _EventStreamFactory = Callable[[], AsyncIterator[AgentEvent]]


class RunObservers:
    """Start passive, failure-isolated consumers for one run."""

    def __init__(self, observers: Sequence[RunObserver] = ()) -> None:
        """Freeze observers and own their active run-consumer tasks."""

        self._observers = tuple(observers)
        self._tasks: set[asyncio.Task[None]] = set()

    def start(
        self,
        *,
        session_id: str,
        run_id: str,
        events: _EventStreamFactory,
    ) -> None:
        """Start and retain every observer with an independent event cursor."""

        for observer in self._observers:
            task = asyncio.create_task(
                _observe(
                    observer,
                    RunEventStream(
                        session_id=session_id,
                        run_id=run_id,
                        events=events(),
                    ),
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)


_LOGGER = logging.getLogger(__name__)


async def _observe(observer: RunObserver, stream: RunEventStream) -> None:
    """Run one observer without letting its failure affect execution."""

    try:
        await observer(stream)
    except Exception:
        _LOGGER.exception(
            "Tile run observer failed for %s/%s.",
            stream.session_id,
            stream.run_id,
        )

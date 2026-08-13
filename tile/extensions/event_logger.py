"""Built-in logging observer extension."""

from __future__ import annotations

import logging

from tile.extensions.registry import ExtensionRegistry
from tile.extensions.run_observers import RunEventStream


class EventLogger:
    """Log every run event observed by one harness."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
    ) -> None:
        """Configure the destination logger and severity level."""

        self._logger = logger or logging.getLogger("tile.events")
        self._level = level

    def register(self, registry: ExtensionRegistry) -> None:
        """Register this extension's passive run observer."""

        registry.observe(self.observe)

    async def observe(self, stream: RunEventStream) -> None:
        """Log every event from one run stream."""

        async for event in stream:
            self._logger.log(
                self._level,
                "Tile run event session_id=%s run_id=%s event=%s",
                stream.session_id,
                stream.run_id,
                event.model_dump(mode="json"),
            )

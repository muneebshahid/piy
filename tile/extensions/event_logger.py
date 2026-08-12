"""Built-in logging observer extension."""

from __future__ import annotations

import logging

from tile.extensions.registry import ExtensionRegistry
from tile.extensions.run_observers import RunEvent


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

    def observe(self, event: RunEvent) -> None:
        """Log one run event with its session and run identities."""

        self._logger.log(
            self._level,
            "Tile run event session_id=%s run_id=%s event=%s",
            event.session_id,
            event.run_id,
            event.event.model_dump(mode="json"),
        )

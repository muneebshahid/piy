"""Provider abstraction used by agent harness execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy

from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.tools import JsonObject, ToolDefinition


class Provider(ABC):
    """Configured model provider capable of starting assistant streams."""

    def __init__(
        self,
        *,
        model: str,
        reasoning: JsonObject | None = None,
    ) -> None:
        """Retain provider-neutral configuration shared by implementations."""

        self._model = model
        self._reasoning = None if reasoning is None else deepcopy(dict(reasoning))

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider identity persisted with each run."""

    @property
    def model(self) -> str:
        """Return the configured model used for every provider request."""

        return self._model

    @property
    def reasoning(self) -> JsonObject | None:
        """Return a defensive copy of the configured reasoning options."""

        return deepcopy(self._reasoning)

    @abstractmethod
    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Start a provider stream from model-visible history."""

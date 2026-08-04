"""Shared streaming contracts between providers and the runtime."""

from collections.abc import AsyncGenerator

from tile.types.stream_events import ProviderStreamEvent

type AsyncEventStream = AsyncGenerator[ProviderStreamEvent]
"""Async stream of provider-originated assistant events.

Provider streams are async generators: the consumer that iterates a
stream owns closing it, and closure must release the underlying
transport. Adapters wrapping SDK streams forward closure to the SDK
object in a ``finally``.
"""

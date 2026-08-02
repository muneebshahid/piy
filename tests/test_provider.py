"""Tests for the configured provider boundary."""

import asyncio

from tile.types import JsonObject
from tests.support.agent_streams import ProviderStreamMock, final_text_stream


def test_provider_exposes_configuration_and_delegates_streaming() -> None:
    """Bind model and reasoning while preserving the transport contract."""

    reasoning: JsonObject = {"effort": "high"}
    provider = ProviderStreamMock(
        [final_text_stream("response-1", "done")],
        model="gpt-5.4",
        reasoning=reasoning,
    )

    async def invoke() -> None:
        """Invoke the configured provider through its stream contract."""

        await provider.stream((), instructions="Test", tools=None)

    asyncio.run(invoke())

    assert provider.name == "test"
    assert provider.reasoning == reasoning
    assert provider.model == "gpt-5.4"

    reasoning["effort"] = "low"
    exposed_reasoning = provider.reasoning
    assert exposed_reasoning is not None
    exposed_reasoning["effort"] = "medium"
    assert provider.reasoning == {"effort": "high"}

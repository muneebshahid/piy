"""Tests for the configured provider boundary."""

from tile.types import JsonObject
from tests.support.agent_streams import ProviderStreamMock


def test_provider_isolates_reasoning_from_caller_and_reader_mutation() -> None:
    """Deep-copy reasoning so caller and reader mutations never leak in."""

    reasoning: JsonObject = {"effort": "high"}
    provider = ProviderStreamMock([], reasoning=reasoning)

    reasoning["effort"] = "low"
    exposed_reasoning = provider.reasoning
    assert exposed_reasoning is not None
    exposed_reasoning["effort"] = "medium"

    assert provider.reasoning == {"effort": "high"}

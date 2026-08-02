"""Tests for the guaranteed run lifecycle across failures and aborts."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, cast
from unittest.mock import AsyncMock

from pydantic import BaseModel

from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    RunStartEvent,
    ToolExecutionStartEvent,
)
from tile import AgentHarness, Provider, SessionRepository
from tile.result import Aborted, Completed, ExecutionFailure, Failed
from tile.store import SQLiteStore
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderStreamEvent
from tile.types.tools import ToolDefinition, ToolFunction, ToolResult
from tests.support.agent_streams import (
    TEST_PROVIDER,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_start,
    tool_call_stream,
)
from tests.support.async_streams import async_stream
from tests.support.tool_definitions import CityInput, city_tool


def assert_run_lifecycle(events: Sequence[AgentEvent]) -> None:
    """Assert the guaranteed run start and terminal run end."""

    assert events, "empty run log"
    assert events[0].type == "run_start"
    assert events[-1].type == "run_end"
    assert sum(event.type == "run_start" for event in events) == 1
    assert sum(event.type == "run_end" for event in events) == 1


class WeatherReport(BaseModel):
    """Sample result schema for typed-result attempt tests."""

    city: str
    temp_c: float


class _ProviderCall(Protocol):
    """Callable shape used by lifecycle-specific provider mocks."""

    def __call__(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> Awaitable[AsyncEventStream]: ...


class _MockProvider(Provider):
    """Configured provider around one lifecycle-specific async mock."""

    def __init__(self, call: _ProviderCall) -> None:
        """Bind the mock call to the deterministic test model."""

        super().__init__(model="gpt-5.4")
        self._call = call

    @property
    def name(self) -> str:
        """Return the deterministic provider identity."""

        return TEST_PROVIDER

    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Delegate provider acquisition to the configured mock."""

        return await self._call(
            history,
            instructions=instructions,
            tools=tools,
        )


def test_provider_raise_before_stream_fails_the_run() -> None:
    """Close the run when the provider dies before streaming."""

    failing_mock = AsyncMock(side_effect=ConnectionError("connection refused"))
    harness, provider = _harness(
        _mock_provider(failing_mock), session_id="raise-before-stream"
    )

    async def _run() -> list[AgentEvent]:
        """Fail the run and collect its complete log."""

        run = await harness.prompt("hello", provider=provider)
        assert isinstance(await run.wait(), Failed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "run_end",
    ]
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.message == "connection refused"


def test_abort_during_provider_acquisition_closes_the_run() -> None:
    """Close the run when cancellation lands before a stream is acquired."""

    async def _run() -> list[AgentEvent]:
        """Start a blocked provider acquisition, cancel it, and collect events."""

        entered = asyncio.Event()

        async def _blocked_provider(
            history: Sequence[ConversationItem],
            *,
            instructions: str,
            tools: Sequence[ToolDefinition] | None,
        ) -> AsyncEventStream:
            """Signal provider entry and block until cancellation."""

            _ = history, instructions, tools
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Blocked provider unexpectedly resumed.")

        blocked_mock = AsyncMock(side_effect=_blocked_provider)
        harness, provider = _harness(
            _mock_provider(blocked_mock), session_id="abort-during-acquisition"
        )
        run = await harness.prompt("hello", provider=provider)
        await entered.wait()
        run.abort()
        assert await run.wait() == Aborted()
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "run_end",
    ]
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_provider_raise_mid_stream_leaves_inner_lifecycles_open() -> None:
    """Let the run end terminate inner lifecycles after a provider failure."""

    interrupted_mock = AsyncMock(
        return_value=async_stream(
            [stream_start("resp_1")], error=ConnectionError("connection reset")
        )
    )
    harness, provider = _harness(
        _mock_provider(interrupted_mock), session_id="raise-mid-stream"
    )

    async def _run() -> list[AgentEvent]:
        """Fail the run mid-message and collect its complete log."""

        run = await harness.prompt("hello", provider=provider)
        assert isinstance(await run.wait(), Failed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    ]
    assert events[-1] == RunEndEvent(
        outcome=Failed(
            cause=ExecutionFailure(
                origin="execution",
                exception_type="ConnectionError",
                message="connection reset",
            )
        )
    )


def test_stream_exhausted_without_terminal_event_leaves_inner_scopes_open() -> None:
    """Let the run end terminate scopes when a provider omits its terminal."""

    quiet_mock = AsyncMock(return_value=async_stream([stream_start("resp_1")]))
    harness, provider = _harness(
        _mock_provider(quiet_mock), session_id="quiet-stream-death"
    )

    async def _run() -> tuple[list[AgentEvent], str | None]:
        """Fail the run on a stream that ends without a terminal event."""

        run = await harness.prompt("hello", provider=provider)
        outcome = await run.wait()
        assert isinstance(outcome, Failed)
        assert isinstance(outcome.cause, ExecutionFailure)
        return [event async for event in run.events()], outcome.cause.message

    events, error_message = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    ]
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "execution"
    assert error_message == (
        "Provider stream ended without StreamDoneEvent or StreamErrorEvent."
    )


def test_in_band_stream_error_ends_message_before_run_end() -> None:
    """Keep the provider-finalized message before the failed run ends."""

    harness, provider = _harness(
        ProviderStreamMock([error_stream("resp_1", "boom")]),
        session_id="in-band-error",
    )

    async def _run() -> list[AgentEvent]:
        """Fail the run through an in-band stream error event."""

        run = await harness.prompt("hello", provider=provider)
        assert isinstance(await run.wait(), Failed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert _single(events, MessageEndEvent).assistant_turn.status == "error"
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"


def test_abort_during_tool_execution_leaves_inner_lifecycles_open() -> None:
    """Let the run end terminate an active tool and its outer lifecycles."""

    async def _blocked(params: CityInput) -> ToolResult:
        """Block forever so the abort lands inside the tool."""

        _ = params
        await asyncio.Event().wait()
        return ToolResult.text("never")

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            )
        ]
    )
    harness, configured_provider = _harness(
        provider,
        session_id="abort-in-tool",
        tools=[_weather_tool(_blocked)],
    )

    async def _run() -> list[AgentEvent]:
        """Abort once the tool execution has started."""

        run = await harness.prompt("check weather", provider=configured_provider)
        async for event in run.events():
            if isinstance(event, ToolExecutionStartEvent):
                break
        run.abort()
        assert await run.wait() == Aborted()
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "run_end",
    ]
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_abort_during_provider_stream_leaves_inner_lifecycles_open() -> None:
    """Let the run end terminate an active message and its outer lifecycles."""

    async def _stalled(
        events: Sequence[ProviderStreamEvent],
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Yield the given events, then stall until cancelled."""

        for event in events:
            yield event
        await asyncio.Event().wait()

    stalled_mock = AsyncMock(return_value=_stalled([stream_start("resp_1")]))
    harness, provider = _harness(
        _mock_provider(stalled_mock), session_id="abort-mid-stream"
    )

    async def _run() -> list[AgentEvent]:
        """Abort once the message has started streaming."""

        run = await harness.prompt("hello", provider=provider)
        async for event in run.events():
            if isinstance(event, MessageStartEvent):
                break
        run.abort()
        assert await run.wait() == Aborted()
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "run_end",
    ]
    assert _single(events, RunEndEvent).outcome == Aborted()


def test_abort_before_first_tick_still_yields_a_closed_log() -> None:
    """Close the run scope for an abort landing before execution starts."""

    provider = ProviderStreamMock([final_text_stream("resp_1", "hello back")])
    harness, configured_provider = _harness(provider, session_id="abort-before-tick")

    async def _run() -> list[AgentEvent]:
        """Abort synchronously after submission, before the first tick."""

        run = await harness.prompt("hello", provider=configured_provider)
        run.abort()
        assert await run.wait() == Aborted()
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert events == [RunStartEvent(), RunEndEvent(outcome=Aborted())]


def test_typed_result_attempts_each_close_before_the_next_starts() -> None:
    """Pair every typed-result attempt sequentially around the follow-up."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Still thinking."),
            tool_call_stream(
                response_id="resp_2",
                call_id="call_1",
                tool_name="complete",
                arguments={"city": "Munich", "temp_c": 21.0},
            ),
        ]
    )
    harness, configured_provider = _harness(provider, session_id="typed-attempts")

    async def _run() -> list[AgentEvent]:
        """Complete the typed result on the nudged second attempt."""

        run = await harness.prompt(
            "Weather?", provider=configured_provider, result=WeatherReport
        )
        assert isinstance(await run.wait(), Completed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert_run_lifecycle(events)
    start_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event, AgentStartEvent)
    ]
    end_indices = [
        index for index, event in enumerate(events) if isinstance(event, AgentEndEvent)
    ]
    assert len(start_indices) == 2
    assert len(end_indices) == 2
    assert end_indices[0] < start_indices[1]


def test_tool_loop_prompt_yields_the_full_expected_event_order() -> None:
    """Pin the complete runtime event order for a tool-use prompt."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_2", "It is sunny in Munich."),
        ]
    )
    harness, configured_provider = _harness(
        provider,
        session_id="full-order",
        tools=[_weather_tool(_quick_weather)],
    )

    async def _run() -> list[AgentEvent]:
        """Complete one tool-loop prompt and collect its full log."""

        run = await harness.prompt("check weather", provider=configured_provider)
        assert isinstance(await run.wait(), Completed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "run_end",
    ]


def test_typed_result_prompt_yields_the_full_expected_event_order() -> None:
    """Pin the complete runtime event order across a nudged typed run."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Still thinking."),
            tool_call_stream(
                response_id="resp_2",
                call_id="call_1",
                tool_name="complete",
                arguments={"city": "Munich", "temp_c": 21.0},
            ),
        ]
    )
    harness, configured_provider = _harness(provider, session_id="typed-full-order")

    async def _run() -> list[AgentEvent]:
        """Complete the typed result on the nudged second attempt."""

        run = await harness.prompt(
            "Weather?", provider=configured_provider, result=WeatherReport
        )
        assert isinstance(await run.wait(), Completed)
        return [event async for event in run.events()]

    events = asyncio.run(_run())

    assert [event.type for event in events] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "result_follow_up",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "agent_end",
        "run_end",
    ]


def _harness(
    provider: Provider,
    *,
    session_id: str,
    tools: Sequence[ToolDefinition] = (),
) -> tuple[AgentHarness, Provider]:
    """Build a session-bound harness and configured test provider."""

    repository = SessionRepository(SQLiteStore(in_memory=True))
    session = repository.create(session_id=session_id)
    harness = AgentHarness(
        session=session,
        cwd=Path("."),
        tools=tools,
    )
    return harness, provider


def _mock_provider(mock: AsyncMock) -> Provider:
    """Configure one lifecycle-specific async mock as a provider."""

    return _MockProvider(cast("_ProviderCall", mock))


def _weather_tool(fn: ToolFunction) -> ToolDefinition:
    """Build the deterministic weather tool around one implementation."""

    return city_tool("get_weather", "Return a deterministic weather report.", fn)


async def _quick_weather(params: CityInput) -> ToolResult:
    """Return deterministic weather text immediately."""

    return ToolResult.text(f"{params.city}: sunny")


_EventT = TypeVar("_EventT", bound=AgentEvent)


def _single(events: Sequence[AgentEvent], event_type: type[_EventT]) -> _EventT:
    """Return the only event of one type in a run log."""

    matches = [event for event in events if isinstance(event, event_type)]
    assert len(matches) == 1, f"expected one {event_type.__name__}, got {len(matches)}"
    return matches[0]

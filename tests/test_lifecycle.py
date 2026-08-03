"""Tests for the guaranteed run lifecycle across failures and aborts."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import NamedTuple, Protocol, override
from unittest.mock import AsyncMock

import pytest

from tests.support.agent_streams import (
    TEST_PROVIDER,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_start,
    tool_call_stream,
)
from tests.support.async_streams import async_stream
from tests.support.harnesses import build_harness
from tests.support.tool_definitions import CityInput, WeatherReport, city_tool
from tile import Provider, RunHandle
from tile.events import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    ToolExecutionStartEvent,
)
from tile.result import Aborted, Completed, ExecutionFailure, Failed
from tile.store import SQLiteStore
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderStreamEvent
from tile.types.tools import ToolDefinition, ToolFunction, ToolResult


def assert_run_lifecycle(events: Sequence[AgentEvent]) -> None:
    """Assert the guaranteed run start and terminal run end."""

    assert events, "empty run log"
    assert events[0].type == "run_start"
    assert events[-1].type == "run_end"
    assert sum(event.type == "run_start" for event in events) == 1
    assert sum(event.type == "run_end" for event in events) == 1


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
    @override
    def name(self) -> str:
        """Return the deterministic provider identity."""

        return TEST_PROVIDER

    @override
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


async def test_provider_raise_before_stream_fails_the_run(
    store: SQLiteStore,
) -> None:
    """Close the run when the provider dies before streaming."""

    failing_mock = AsyncMock(side_effect=ConnectionError("connection refused"))
    harness = build_harness(store, session_id="raise-before-stream")

    run = await harness.prompt("hello", provider=_mock_provider(failing_mock))
    assert isinstance(await run.wait(), Failed)
    events = [event async for event in run.events()]

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


@pytest.mark.parametrize(
    ("stream_error", "expected_exception_type", "expected_message"),
    [
        pytest.param(
            ConnectionError("connection reset"),
            "ConnectionError",
            "connection reset",
            id="mid-stream-raise",
        ),
        pytest.param(
            None,
            "ProviderStreamProtocolError",
            "Provider stream ended without StreamDoneEvent or StreamErrorEvent.",
            id="exhausted-without-terminal",
        ),
    ],
)
async def test_stream_death_leaves_inner_lifecycles_open(
    store: SQLiteStore,
    stream_error: Exception | None,
    expected_exception_type: str,
    expected_message: str,
) -> None:
    """Let the run end terminate inner lifecycles when the stream dies."""

    dead_mock = AsyncMock(
        return_value=async_stream([stream_start("resp_1")], error=stream_error)
    )
    harness = build_harness(store, session_id="stream-death")

    run = await harness.prompt("hello", provider=_mock_provider(dead_mock))
    assert isinstance(await run.wait(), Failed)
    events = [event async for event in run.events()]

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
                exception_type=expected_exception_type,
                message=expected_message,
            )
        )
    )


async def test_in_band_stream_error_ends_message_before_run_end(
    store: SQLiteStore,
) -> None:
    """Keep the provider-finalized message before the failed run ends."""

    provider = ProviderStreamMock([error_stream("resp_1", "boom")])
    harness = build_harness(store, session_id="in-band-error")

    run = await harness.prompt("hello", provider=provider)
    assert isinstance(await run.wait(), Failed)
    events = [event async for event in run.events()]

    assert_run_lifecycle(events)
    assert _single(events, MessageEndEvent).assistant_turn.status == "error"
    run_outcome = _single(events, RunEndEvent).outcome
    assert isinstance(run_outcome, Failed)
    assert isinstance(run_outcome.cause, ExecutionFailure)
    assert run_outcome.cause.origin == "turn"


class _AbortSetup(NamedTuple):
    """Provider, tools, and abort trigger for one abort landing point."""

    provider: Provider
    tools: Sequence[ToolDefinition]
    ready: Callable[[RunHandle], Awaitable[None]]


async def _no_wait(run: RunHandle) -> None:
    """Let the abort land synchronously, before the first scheduler tick."""

    _ = run


def _wait_for_event(
    event_type: type[AgentEvent],
) -> Callable[[RunHandle], Awaitable[None]]:
    """Build a trigger that waits until one live event type is observed."""

    async def _wait(run: RunHandle) -> None:
        """Consume live events until the trigger event arrives."""

        async for event in run.events():
            if isinstance(event, event_type):
                return

    return _wait


def _abort_before_tick_setup() -> _AbortSetup:
    """Pair a completing stream with an abort that preempts execution."""

    provider = ProviderStreamMock([final_text_stream("resp_1", "hello back")])
    return _AbortSetup(provider=provider, tools=(), ready=_no_wait)


def _abort_in_acquisition_setup() -> _AbortSetup:
    """Block provider acquisition so the abort lands before any stream."""

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

    async def _ready(run: RunHandle) -> None:
        """Wait until the blocked provider call has been entered."""

        _ = run
        await entered.wait()

    provider = _mock_provider(AsyncMock(side_effect=_blocked_provider))
    return _AbortSetup(provider=provider, tools=(), ready=_ready)


def _abort_in_stream_setup() -> _AbortSetup:
    """Stall the provider stream so the abort lands mid-message."""

    async def _stalled() -> AsyncIterator[ProviderStreamEvent]:
        """Yield the stream start, then stall until cancelled."""

        yield stream_start("resp_1")
        await asyncio.Event().wait()

    provider = _mock_provider(AsyncMock(return_value=_stalled()))
    return _AbortSetup(
        provider=provider,
        tools=(),
        ready=_wait_for_event(MessageStartEvent),
    )


def _abort_in_tool_setup() -> _AbortSetup:
    """Block tool execution so the abort lands inside the tool."""

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
    return _AbortSetup(
        provider=provider,
        tools=(_weather_tool(_blocked_weather),),
        ready=_wait_for_event(ToolExecutionStartEvent),
    )


@pytest.mark.parametrize(
    ("make_setup", "expected_event_types"),
    [
        pytest.param(
            _abort_before_tick_setup,
            ["run_start", "run_end"],
            id="before-first-tick",
        ),
        pytest.param(
            _abort_in_acquisition_setup,
            ["run_start", "agent_start", "run_end"],
            id="during-provider-acquisition",
        ),
        pytest.param(
            _abort_in_stream_setup,
            ["run_start", "agent_start", "turn_start", "message_start", "run_end"],
            id="during-provider-stream",
        ),
        pytest.param(
            _abort_in_tool_setup,
            [
                "run_start",
                "agent_start",
                "turn_start",
                "message_start",
                "message_end",
                "tool_execution_start",
                "run_end",
            ],
            id="during-tool-execution",
        ),
    ],
)
async def test_abort_closes_the_run_from_any_landing_point(
    store: SQLiteStore,
    make_setup: Callable[[], _AbortSetup],
    expected_event_types: list[str],
) -> None:
    """Close the run scope for an abort landing at any execution point."""

    setup = make_setup()
    harness = build_harness(store, session_id="abort-landing", tools=setup.tools)

    run = await harness.prompt("hello", provider=setup.provider)
    await setup.ready(run)
    run.abort()
    assert await run.wait() == Aborted()
    events = [event async for event in run.events()]

    assert_run_lifecycle(events)
    assert [event.type for event in events] == expected_event_types
    assert _single(events, RunEndEvent).outcome == Aborted()


async def test_tool_loop_prompt_yields_the_full_expected_event_order(
    store: SQLiteStore,
) -> None:
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
    harness = build_harness(
        store,
        session_id="full-order",
        tools=[_weather_tool(_quick_weather)],
    )

    run = await harness.prompt("check weather", provider=provider)
    assert isinstance(await run.wait(), Completed)
    events = [event async for event in run.events()]

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


async def test_typed_result_prompt_yields_the_full_expected_event_order(
    store: SQLiteStore,
) -> None:
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
    harness = build_harness(store, session_id="typed-full-order")

    run = await harness.prompt("Weather?", provider=provider, result=WeatherReport)
    assert isinstance(await run.wait(), Completed)
    events = [event async for event in run.events()]

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


def _mock_provider(mock: AsyncMock) -> Provider:
    """Configure one lifecycle-specific async mock as a provider."""

    return _MockProvider(mock)


def _weather_tool(fn: ToolFunction) -> ToolDefinition:
    """Build the deterministic weather tool around one implementation."""

    return city_tool("get_weather", "Return a deterministic weather report.", fn)


async def _quick_weather(params: CityInput) -> ToolResult:
    """Return deterministic weather text immediately."""

    return ToolResult.text(f"{params.city}: sunny")


async def _blocked_weather(params: CityInput) -> ToolResult:
    """Block forever so the abort lands inside the tool."""

    _ = params
    await asyncio.Event().wait()
    return ToolResult.text("never")


def _single[EventT: AgentEvent](
    events: Sequence[AgentEvent], event_type: type[EventT]
) -> EventT:
    """Return the only event of one type in a run log."""

    matches = [event for event in events if isinstance(event, event_type)]
    assert len(matches) == 1, f"expected one {event_type.__name__}, got {len(matches)}"
    return matches[0]

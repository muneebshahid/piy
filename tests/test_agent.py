"""Tests for translating stream events into agent events.

These tests document the final streaming lifecycle layer. The agent receives
app-level ``StreamEvent`` models, emits ``message_start`` and ``message_update``
events while the assistant message is streaming, finalizes history on ``done``
or ``error``, and executes tools before starting a follow-up assistant turn.
"""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import pytest

from tile.agent import AgentResult, run_agent
from tile.exceptions import (
    ProviderStreamProtocolError,
    TurnFailedError,
)
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tile.providers.base import Provider
from tile.tool_executor import ToolExecutor
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import (
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import (
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    ProviderStreamEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from tile.types.tool_execution import ToolInputValidationFailure, ToolInvocationFailure
from tile.tools.read import ReadDetails
from tile.tool_truncation import ToolOutputDetails
from tile.types.tools import (
    ToolDefinition,
    ToolInput,
    ToolResult,
)
from tests.support.agent_streams import (
    ProviderStreamMock,
    empty_stream,
    error_stream,
    final_text_stream,
    stream_done,
    stream_start,
    tool_call_block,
    tool_call_stream,
)
from tests.support.agent_runs import collect_agent_run, collect_run_events
from tests.support.conversation_assertions import (
    expect_assistant_turn,
    expect_tool_result_turn,
    expect_user_message,
)
from tests.support.stream_assertions import expect_stream_event
from tests.support.tool_definitions import CityInput, city_tool
from tests.support.tool_results import tool_text

TEvent = TypeVar("TEvent", bound=AgentEvent)


class ClosingProvider(Provider):
    """Provider fake that records deterministic stream closure."""

    def __init__(self) -> None:
        """Create an open provider stream."""

        super().__init__(model="gpt-5.4")
        self.closed = False

    @property
    def name(self) -> str:
        """Return the deterministic provider identity."""

        return "test"

    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return a stream whose finalizer records closure."""

        _ = history, instructions, tools
        return self._events()

    async def _events(self) -> AsyncEventStream:
        """Emit one completed turn and record generator closure."""

        try:
            yield stream_start("resp_close")
            yield stream_done("resp_close")
        finally:
            self.closed = True


@dataclass(frozen=True)
class ToolUseLoopRun:
    """Captured events and expected blocks for the weather tool-loop scenario."""

    events: list[AgentEvent]
    result: AgentResult
    provider: ProviderStreamMock
    tools: list[ToolDefinition]
    reasoning_block: ReasoningBlock
    tool_call_block: ToolCallBlock
    text_block: TextBlock


def _expect_event_type(event: AgentEvent, event_type: type[TEvent]) -> TEvent:
    """Assert and return an agent event with a precise type."""

    assert isinstance(event, event_type)
    return event


def _sample_tools() -> list[ToolDefinition]:
    """Build the deterministic tool registry used by agent tests."""

    return [
        city_tool(
            "get_weather",
            "Return a simple weather report for a city.",
            _get_weather,
        )
    ]


async def _get_weather(params: CityInput) -> ToolResult:
    """Return a deterministic weather payload for tests."""

    return ToolResult.text(
        json.dumps(
            {
                "temperature_c": 18,
                "condition": "sunny",
                "city": params.city,
            }
        )
    )


async def _raise_tool_error(params: CityInput) -> ToolResult:
    """Raise a deterministic tool error for tests."""

    _ = params
    raise RuntimeError("boom")


async def _read_file(params: ToolInput) -> ToolResult:
    """Return a deterministic file read result with runtime metadata."""

    _ = params
    return ToolResult.text(
        "file contents",
        details=ReadDetails(output=_tool_output_details()),
    )


def _tool_output_details() -> ToolOutputDetails:
    """Build deterministic bounded-output metadata for tool tests."""

    return ToolOutputDetails(
        truncated=False,
        truncated_by=None,
        keep="head",
        total_lines=1,
        total_bytes=13,
        output_lines=1,
        output_bytes=13,
        edge_line_exceeds_limit=False,
        max_lines=100,
        max_bytes=1024,
    )


def _collect_weather_tool_loop_run() -> ToolUseLoopRun:
    """Run a two-turn weather tool loop and capture its observable state."""

    tools = _sample_tools()
    reasoning_block = ReasoningBlock(summary_text="Thinking about weather")
    weather_tool_call_block = tool_call_block(
        call_id="call_123",
        name="get_weather",
        arguments={"city": "Munich"},
        provider_item_id="fc_123",
    )
    text_block = TextBlock(text="It is sunny in Munich.")
    provider = ProviderStreamMock(
        [
            _weather_tool_call_stream(reasoning_block, weather_tool_call_block),
            _weather_follow_up_stream(text_block),
        ]
    )
    history: list[ConversationItem] = [
        UserMessage(content="What is the weather in Munich?")
    ]

    capture = collect_agent_run(
        history,
        provider=provider,
        tools=tools,
    )
    return ToolUseLoopRun(
        events=capture.events,
        result=capture.result,
        provider=provider,
        tools=tools,
        reasoning_block=reasoning_block,
        tool_call_block=weather_tool_call_block,
        text_block=text_block,
    )


def _weather_tool_call_stream(
    reasoning_block: ReasoningBlock,
    weather_tool_call_block: ToolCallBlock,
) -> list[ProviderStreamEvent]:
    """Build a provider stream that requests the weather tool with deltas."""

    return [
        stream_start("resp_tool_call"),
        ReasoningStartEvent(content_index=0),
        ReasoningDeltaEvent(content_index=0, delta="Thinking about weather"),
        ReasoningEndEvent(content_index=0, block=reasoning_block),
        ToolCallStartEvent(
            content_index=1,
            call_id="call_123",
            name="get_weather",
        ),
        ToolCallDeltaEvent(content_index=1, delta='{"city":"Munich"}'),
        ToolCallEndEvent(content_index=1, block=weather_tool_call_block),
        stream_done(
            "resp_tool_call",
            stop_reason="tool_use",
            blocks=[weather_tool_call_block],
        ),
    ]


def _weather_follow_up_stream(
    text_block: TextBlock,
) -> list[ProviderStreamEvent]:
    """Build a provider stream that answers after tool execution."""

    return [
        stream_start("resp_follow_up"),
        TextStartEvent(content_index=0),
        TextDeltaEvent(content_index=0, delta="It is sunny in Munich."),
        TextEndEvent(content_index=0, block=text_block),
        stream_done("resp_follow_up", blocks=[text_block]),
    ]


def test_run_agent_does_not_mutate_supplied_history() -> None:
    """Keep caller-owned history unchanged while emitting stable events."""

    provider = ProviderStreamMock(
        [
            empty_stream("resp_done"),
        ]
    )
    history: list[ConversationItem] = [UserMessage(content="Hello, Tile")]

    events = collect_run_events(history, provider=provider)

    message_end = _expect_event_type(events[3], MessageEndEvent)
    _expect_event_type(events[-1], AgentEndEvent)
    assert history == [UserMessage(content="Hello, Tile")]
    assert message_end.assistant_turn.response_id == "resp_done"


def test_run_agent_returns_terminal_result() -> None:
    """Return the last assistant turn independently from emitted events."""

    provider = ProviderStreamMock([final_text_stream("resp_done", "Done")])

    async def _run() -> AgentResult:
        """Run the agent while discarding its observable events."""

        return await run_agent(
            [UserMessage(content="Hello")],
            emit=lambda _event: None,
            provider=provider,
            tool_executor=ToolExecutor(()),
            instructions="Base prompt.",
        )

    result = asyncio.run(_run())

    assert result.last_turn.response_id == "resp_done"
    assert result.tool_executions == ()


def test_run_agent_returns_accumulated_tool_executions() -> None:
    """Return tool outcomes accumulated across a multi-turn tool loop."""

    run = _collect_weather_tool_loop_run()

    assert run.result.last_turn.response_id == "resp_follow_up"
    assert len(run.result.tool_executions) == 1
    tool_execution = run.result.tool_executions[0]
    assert run.result.tool_executions == (tool_execution,)
    assert tool_execution.tool_result_turn.call_id == "call_123"
    assert tool_execution.tool_result_turn.tool_name == "get_weather"
    assert tool_text(tool_execution.tool_result_turn) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )


def test_run_agent_closes_provider_stream_when_emit_fails() -> None:
    """Close the acquired provider stream when event emission raises."""

    provider = ClosingProvider()

    def emit(event: AgentEvent) -> None:
        """Fail after the provider stream has started."""

        if isinstance(event, MessageStartEvent):
            raise RuntimeError("event sink unavailable")

    async def _run() -> None:
        """Run the agent through the failing event sink."""

        with pytest.raises(RuntimeError, match="event sink unavailable"):
            await run_agent(
                [UserMessage(content="Hello")],
                emit=emit,
                provider=provider,
                tool_executor=ToolExecutor(()),
                instructions="Base prompt.",
            )

    asyncio.run(_run())

    assert provider.closed


def test_empty_provider_stream_fails_after_agent_start() -> None:
    """Reject an empty provider stream before opening a turn or message."""

    provider = ProviderStreamMock([[]])
    events: list[AgentEvent] = []

    async def _run() -> None:
        """Run one empty provider stream."""

        with pytest.raises(
            ProviderStreamProtocolError,
            match="ended without StreamDoneEvent or StreamErrorEvent",
        ):
            await run_agent(
                [UserMessage(content="Hello")],
                emit=events.append,
                provider=provider,
                tool_executor=ToolExecutor(()),
                instructions="Base prompt.",
            )

    asyncio.run(_run())

    assert [event.type for event in events] == ["agent_start"]


def test_stream_exhaustion_leaves_message_and_turn_open() -> None:
    """Reject a started provider stream that omits its terminal event."""

    provider = ProviderStreamMock([[stream_start("resp_quiet")]])
    events: list[AgentEvent] = []

    async def _run() -> None:
        """Run one unterminated provider stream."""

        with pytest.raises(ProviderStreamProtocolError):
            await run_agent(
                [UserMessage(content="Hello")],
                emit=events.append,
                provider=provider,
                tool_executor=ToolExecutor(()),
                instructions="Base prompt.",
            )

    asyncio.run(_run())

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
    ]


def test_agent_run_emits_expected_event_sequence_for_tool_use_loop() -> None:
    """Emit the expected event sequence for a tool-use loop."""

    run = _collect_weather_tool_loop_run()

    assert [event.type for event in run.events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]


def test_agent_run_emits_current_tool_use_stream_events() -> None:
    """Forward first-turn reasoning and tool-call stream events."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    first_message_start = _expect_event_type(events[2], MessageStartEvent)
    first_reasoning_start = _expect_event_type(events[3], MessageUpdateEvent)
    first_reasoning_delta = _expect_event_type(events[4], MessageUpdateEvent)
    first_reasoning_end = _expect_event_type(events[5], MessageUpdateEvent)
    first_tool_call_start = _expect_event_type(events[6], MessageUpdateEvent)
    first_tool_call_delta = _expect_event_type(events[7], MessageUpdateEvent)
    first_tool_call_end = _expect_event_type(events[8], MessageUpdateEvent)
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert first_message_start.response_id == "resp_tool_call"
    assert first_reasoning_start.stream_event.type == "reasoning_start"
    assert first_reasoning_start.stream_event.content_index == 0
    assert first_reasoning_delta.stream_event.type == "reasoning_delta"
    assert (
        expect_stream_event(
            first_reasoning_delta.stream_event, ReasoningDeltaEvent
        ).delta
        == "Thinking about weather"
    )
    assert first_reasoning_end.stream_event.type == "reasoning_end"
    assert (
        expect_stream_event(first_reasoning_end.stream_event, ReasoningEndEvent).block
        == run.reasoning_block
    )
    assert first_tool_call_start.stream_event.type == "tool_call_start"
    assert first_tool_call_start.stream_event.content_index == 1
    assert first_tool_call_delta.stream_event.type == "tool_call_delta"
    assert (
        expect_stream_event(
            first_tool_call_delta.stream_event, ToolCallDeltaEvent
        ).delta
        == '{"city":"Munich"}'
    )
    assert first_tool_call_end.stream_event.type == "tool_call_end"
    assert (
        expect_stream_event(first_tool_call_end.stream_event, ToolCallEndEvent).block
        == run.tool_call_block
    )


def test_agent_run_emits_current_follow_up_stream_events() -> None:
    """Forward second-turn text stream events after tool execution."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    second_message_start = _expect_event_type(events[14], MessageStartEvent)
    second_text_start = _expect_event_type(events[15], MessageUpdateEvent)
    second_text_delta = _expect_event_type(events[16], MessageUpdateEvent)
    second_text_end = _expect_event_type(events[17], MessageUpdateEvent)
    _expect_event_type(events[20], AgentEndEvent)

    assert isinstance(events[13], TurnStartEvent)
    assert second_message_start.response_id == "resp_follow_up"
    assert second_text_start.stream_event.type == "text_start"
    assert second_text_start.stream_event.content_index == 0
    assert second_text_delta.stream_event.type == "text_delta"
    assert expect_stream_event(
        second_text_delta.stream_event, TextDeltaEvent
    ).delta == ("It is sunny in Munich.")
    assert second_text_end.stream_event.type == "text_end"
    assert expect_stream_event(second_text_end.stream_event, TextEndEvent).block == (
        run.text_block
    )


def test_agent_run_emits_tool_execution_outcome_for_tool_use_loop() -> None:
    """Emit tool execution details and attach them to the completed turn."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    first_message_end = _expect_event_type(events[9], MessageEndEvent)
    tool_execution_start = _expect_event_type(events[10], ToolExecutionStartEvent)
    tool_execution_end = _expect_event_type(events[11], ToolExecutionEndEvent)
    first_turn_end = _expect_event_type(events[12], TurnEndEvent)
    first_final_message = first_message_end.assistant_turn

    assert first_final_message.response_id == "resp_tool_call"
    assert first_final_message.stop_reason == "tool_use"
    assert tool_execution_start.call_id == "call_123"
    assert tool_execution_start.tool_name == "get_weather"
    assert tool_execution_start.arguments == {"city": "Munich"}
    tool_execution_outcome = tool_execution_end.outcome
    tool_result_turn = tool_execution_outcome.tool_result_turn
    assert tool_result_turn.call_id == "call_123"
    assert tool_result_turn.tool_name == "get_weather"
    assert tool_text(tool_result_turn) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_result_turn.is_error is False
    assert first_turn_end.assistant_turn.response_id == "resp_tool_call"
    assert first_turn_end.assistant_turn.stop_reason == "tool_use"
    assert first_turn_end.assistant_turn.status == "completed"
    assert first_turn_end.assistant_turn.blocks == [run.tool_call_block]
    assert first_turn_end.tool_executions == (tool_execution_outcome,)
    assert first_turn_end.tool_result_turns == (tool_result_turn,)
    first_turn_outcome = first_turn_end.tool_executions[0]
    assert first_turn_outcome.tool_result_turn.call_id == "call_123"
    assert first_turn_outcome.tool_result_turn.tool_name == "get_weather"
    assert first_turn_outcome.tool_result_turn.content == tool_result_turn.content
    assert first_turn_outcome.tool_result_turn.is_error is False


def test_agent_run_sends_tool_result_history_to_follow_up_turn() -> None:
    """Send assistant and tool result turns into the follow-up model request."""

    run = _collect_weather_tool_loop_run()
    events = run.events

    second_message_end = _expect_event_type(events[18], MessageEndEvent)
    second_turn_end = _expect_event_type(events[19], TurnEndEvent)
    second_final_message = second_message_end.assistant_turn

    assert second_final_message.response_id == "resp_follow_up"
    assert second_final_message.stop_reason == "stop"
    assert second_turn_end.assistant_turn.response_id == "resp_follow_up"
    assert second_turn_end.assistant_turn.stop_reason == "stop"
    assert second_turn_end.assistant_turn.blocks == [
        TextBlock(text="It is sunny in Munich.")
    ]
    assert second_turn_end.tool_executions == ()

    assert run.provider.await_count == 2
    assert run.provider.model == "gpt-5.4"
    assert run.provider.tools(0) == tuple(run.tools)
    assert run.provider.tools(1) == tuple(run.tools)

    first_request_history = run.provider.history(0)
    assert expect_user_message(first_request_history[0]).content == (
        "What is the weather in Munich?"
    )

    follow_up_request_history = run.provider.history(1)
    assert expect_assistant_turn(follow_up_request_history[1]).response_id == (
        "resp_tool_call"
    )
    assert expect_tool_result_turn(follow_up_request_history[2]).tool_name == (
        "get_weather"
    )


def test_agent_run_executes_registered_tool_definition() -> None:
    """Execute a registered tool and expose its result through agent events."""

    tools = _sample_tools()
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_123",
                tool_name="get_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_123",
            ),
            empty_stream("resp_follow_up"),
        ]
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = collect_run_events(
        history,
        provider=provider,
        tools=tools,
    )

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    assert tool_text(tool_execution_end.outcome.tool_result_turn) == (
        '{"temperature_c": 18, "condition": "sunny", "city": "Munich"}'
    )
    assert tool_execution_end.outcome.tool_result_turn.is_error is False


def test_agent_run_exposes_tool_details_outside_replay_turn() -> None:
    """Expose non-replay tool details through execution outcomes."""

    class NoInput(ToolInput):
        """Strict empty input for the deterministic read tool."""

    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a deterministic file.",
            input_model=NoInput,
            fn=_read_file,
        )
    ]
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_read",
                tool_name="read_file",
                arguments={},
                provider_item_id="fc_call_read",
            ),
            final_text_stream(
                response_id="resp_follow_up",
                text="I read the file.",
            ),
        ]
    )
    history = [UserMessage(content="Read a file")]

    events = collect_run_events(
        history,
        provider=provider,
        tools=tools,
    )

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    turn_end = _expect_event_type(events[6], TurnEndEvent)
    outcome = tool_execution_end.outcome
    turn_outcome = turn_end.tool_executions[0]
    assert outcome.details == ReadDetails(output=_tool_output_details())
    assert turn_outcome.details == outcome.details

    follow_up_request_history = provider.history(1)
    tool_result_turn = expect_tool_result_turn(follow_up_request_history[2])
    assert tool_result_turn == outcome.tool_result_turn
    assert "details" not in ToolResultTurn.model_fields


def test_agent_run_continues_after_tool_execution_error() -> None:
    """Return error tool results to the model and continue the run."""

    failing_tool = city_tool(
        "fail_weather",
        "Raise a deterministic weather failure.",
        _raise_tool_error,
    )
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_tool_call",
                call_id="call_123",
                tool_name="fail_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_call_123",
            ),
            final_text_stream(
                response_id="resp_follow_up",
                text="The tool failed.",
            ),
        ]
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = collect_run_events(
        history,
        provider=provider,
        tools=[failing_tool],
    )

    tool_execution_end = _expect_event_type(events[5], ToolExecutionEndEvent)
    _expect_event_type(events[-1], AgentEndEvent)
    assert tool_execution_end.outcome.tool_result_turn.is_error is True
    assert tool_text(tool_execution_end.outcome.tool_result_turn) == "boom"
    assert tool_execution_end.outcome.tool_result_turn.tool_name == "fail_weather"
    assert isinstance(tool_execution_end.outcome.details, ToolInvocationFailure)

    follow_up_request_history = provider.history(1)
    tool_result = expect_tool_result_turn(follow_up_request_history[2])
    assert tool_result.is_error is True
    assert tool_result.tool_name == "fail_weather"


def test_agent_allows_model_to_correct_invalid_tool_arguments() -> None:
    """Persist an input error, then execute the model's corrected tool call."""

    calls: list[str] = []

    async def weather(params: CityInput) -> ToolResult:
        """Record valid tool invocations for the correction scenario."""

        calls.append(params.city)
        return ToolResult.text(f"{params.city}: sunny")

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_invalid",
                call_id="call_invalid",
                tool_name="get_weather",
                arguments={"city": 5},
            ),
            tool_call_stream(
                response_id="resp_valid",
                call_id="call_valid",
                tool_name="get_weather",
                arguments={"city": "Munich"},
            ),
            final_text_stream("resp_done", "The weather is sunny."),
        ]
    )
    history = [UserMessage(content="What is the weather in Munich?")]

    events = collect_run_events(
        history,
        provider=provider,
        tools=[city_tool("get_weather", "Return the weather.", weather)],
    )

    executions = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert calls == ["Munich"]
    assert [event.outcome.tool_result_turn.is_error for event in executions] == [
        True,
        False,
    ]
    assert isinstance(executions[0].outcome.details, ToolInputValidationFailure)
    retry_result = expect_tool_result_turn(provider.history(1)[-1])
    assert retry_result.is_error is True
    assert "Invalid arguments for tool 'get_weather'" in tool_text(
        executions[0].outcome.tool_result_turn
    )
    corrected_result = expect_tool_result_turn(provider.history(2)[-1])
    assert corrected_result.is_error is False
    assert "details" not in ToolResultTurn.model_fields


def test_agent_run_handles_multiple_tool_use_turns() -> None:
    """Carry cumulative run-local history through repeated tool turns."""

    tools = _sample_tools()
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_tool_call_1",
                call_id="call_1",
                tool_name="get_weather",
                arguments={"city": "Munich"},
                provider_item_id="fc_call_1",
            ),
            tool_call_stream(
                response_id="resp_tool_call_2",
                call_id="call_2",
                tool_name="get_weather",
                arguments={"city": "Berlin"},
                provider_item_id="fc_call_2",
            ),
            final_text_stream(response_id="resp_final", text="Both cities are sunny."),
        ]
    )
    history = [UserMessage(content="Compare Munich and Berlin weather.")]

    events = collect_run_events(
        history,
        provider=provider,
        tools=tools,
    )

    _expect_event_type(events[-1], AgentEndEvent)
    assert provider.await_count == 3

    second_tool_request_history = provider.history(1)
    assert len(second_tool_request_history) == 3

    final_request_history = provider.history(2)
    assert len(final_request_history) == 5
    assert expect_assistant_turn(final_request_history[1]).response_id == (
        "resp_tool_call_1"
    )
    assert expect_tool_result_turn(final_request_history[2]).call_id == "call_1"
    assert expect_assistant_turn(final_request_history[3]).response_id == (
        "resp_tool_call_2"
    )
    assert expect_tool_result_turn(final_request_history[4]).call_id == "call_2"


def test_agent_sends_instructions_to_the_provider_verbatim() -> None:
    """Pass the caller's resolved system prompt through without composition."""

    provider = ProviderStreamMock(
        [
            empty_stream("resp_done"),
        ]
    )
    history = [UserMessage(content="Hello")]

    collect_run_events(
        history,
        provider=provider,
        instructions="Resolved system prompt.",
    )

    assert provider.instructions() == "Resolved system prompt."


def test_stream_error_ends_message_then_raises_turn_failure() -> None:
    """Retain the provider-finalized error message before failing the turn."""

    provider = ProviderStreamMock(
        [
            error_stream("resp_error", "Socket closed"),
        ]
    )
    history = [UserMessage(content="Say hello")]

    events: list[AgentEvent] = []

    async def _run() -> TurnFailedError:
        """Run until the provider-finalized turn fails."""

        with pytest.raises(TurnFailedError, match="Socket closed") as captured:
            await run_agent(
                history,
                emit=events.append,
                provider=provider,
                tool_executor=ToolExecutor(()),
                instructions="Base prompt.",
            )
        return captured.value

    error = asyncio.run(_run())

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
    ]

    message_start = _expect_event_type(events[2], MessageStartEvent)
    message_end = _expect_event_type(events[3], MessageEndEvent)
    final_message = message_end.assistant_turn

    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert message_start.response_id == "resp_error"
    assert final_message.response_id == "resp_error"
    assert final_message.status == "error"
    assert error.turn == final_message

    provider.assert_awaited_once()
    request_history = provider.history(0)
    assert expect_user_message(request_history[0]).content == "Say hello"

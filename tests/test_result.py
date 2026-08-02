"""Tests for output-contract tools and runtime-owned result enforcement."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from tile import AgentHarness, Provider, SessionRepository
from tile.result import (
    MAX_RESULT_FOLLOW_UPS,
    NO_RESULT_REASON,
    RESULT_CONTRACT,
    RESULT_FOLLOW_UP,
    AgentFailure,
    Completed,
    ExecutionFailure,
    Failed,
)
from tile.store import SQLiteStore
from tile.tool_executor import ToolExecutor
from tile.tools.complete import CompleteDetails, tool as complete_tool
from tile.tools.fail import tool as fail_tool
from tile.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ResultFollowUpEvent,
    ToolExecutionEndEvent,
)
from tile.types.conversation import AssistantTurn, ToolResultTurn, UserMessage
from tile.types.stream_events import TextBlock
from tile.types.tools import ToolDefinition, ToolResult
from tests.support.agent_streams import (
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    stream_done,
    stream_start,
    tool_call_block,
    tool_call_stream,
)
from tests.support.agent_runs import collect_run_events
from tests.support.tool_definitions import CityInput, city_tool


class WeatherReport(BaseModel):
    """Sample result schema used across output contract tests."""

    city: str
    temp_c: float


def _result_tools() -> list[ToolDefinition]:
    """Build the result tool pair for the sample schema."""

    return [complete_tool(WeatherReport), fail_tool]


async def _weather(params: CityInput) -> ToolResult:
    """Return a visible result for same-batch execution assertions."""

    return ToolResult.text(f"Weather retrieved for {params.city}.")


def _agent_end_event(events: Sequence[AgentEvent]) -> AgentEndEvent:
    """Return the terminal agent end event of a collected run."""

    event = events[-1]
    assert isinstance(event, AgentEndEvent)
    return event


def _complete_call_stream(
    response_id: str,
    call_id: str,
    arguments: dict,
) -> list:
    """Build a provider stream that calls the complete result tool."""

    return tool_call_stream(
        response_id=response_id,
        call_id=call_id,
        tool_name="complete",
        arguments=arguments,
    )


def test_complete_tool_validates_and_succeeds() -> None:
    """Accept schema-conforming arguments on the complete tool."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": 21.0},
        )
    )

    assert not outcome.tool_result_turn.is_error
    assert outcome.terminate


def test_complete_tool_rejects_invalid_arguments() -> None:
    """Return a tool error for arguments that violate the result schema."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert outcome.tool_result_turn.is_error
    assert not outcome.terminate


def test_fail_tool_requires_string_reason() -> None:
    """Reject non-string reasons on the fail tool."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="fail",
            arguments={"reason": 5},
        )
    )

    assert outcome.tool_result_turn.is_error
    assert not outcome.terminate


def test_complete_tool_schema_reflects_model_config() -> None:
    """Emit closed schemas only when the result model forbids extras."""

    class OpenReport(BaseModel):
        city: str

    class ClosedReport(BaseModel):
        model_config = ConfigDict(extra="forbid")
        city: str

    open_schema = complete_tool(OpenReport).input_schema
    closed_schema = complete_tool(ClosedReport).input_schema

    assert "additionalProperties" not in open_schema
    assert closed_schema["additionalProperties"] is False


def test_complete_tool_applies_field_defaults() -> None:
    """Fill omitted optional fields from the result model's defaults."""

    class ReportWithDefault(BaseModel):
        city: str
        note: str = "n/a"

    executor = ToolExecutor([complete_tool(ReportWithDefault), fail_tool])

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert not outcome.tool_result_turn.is_error
    details = outcome.details
    assert isinstance(details, CompleteDetails)
    assert details.value == ReportWithDefault(city="Munich", note="n/a")


def test_complete_tool_preserves_aliased_result_fields() -> None:
    """Complete typed runs with provider-visible Pydantic aliases."""

    class AliasedReport(BaseModel):
        """Result contract whose provider field differs from its Python name."""

        city_name: str = Field(alias="city")

    executor = ToolExecutor([complete_tool(AliasedReport), fail_tool])

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich"},
        )
    )

    assert not outcome.tool_result_turn.is_error
    assert outcome.terminate
    details = outcome.details
    assert isinstance(details, CompleteDetails)
    assert details.value == AliasedReport(city="Munich")


def test_complete_tool_returns_validated_value_in_details() -> None:
    """Carry the validated result instance on the execution details."""

    executor = ToolExecutor(_result_tools())

    outcome = asyncio.run(
        executor.execute(
            call_id="call_1",
            tool_name="complete",
            arguments={"city": "Munich", "temp_c": "21.5"},
        )
    )

    details = outcome.details
    assert isinstance(details, CompleteDetails)
    assert details.value == WeatherReport(city="Munich", temp_c=21.5)


def test_completed_round_trips_value_as_plain_data() -> None:
    """Deserialize a serialized outcome into plain data, losing nothing."""

    outcome = Completed(value=WeatherReport(city="Munich", temp_c=21.0))

    revalidated = Completed.model_validate_json(outcome.model_dump_json())

    assert revalidated.value == {"city": "Munich", "temp_c": 21.0}


def test_tool_executor_rejects_duplicate_names() -> None:
    """Refuse to register two tools with the same name."""

    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolExecutor([fail_tool, fail_tool])


def test_agent_stops_after_terminating_tool_batch(tmp_path: Path) -> None:
    """Exit the generic agent loop without another provider call after termination."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    events = collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        provider=provider,
        tools=_result_tools(),
    )

    assert provider.await_count == 1
    executions = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert len(executions) == 1
    assert executions[0].outcome.terminate
    _agent_end_event(events)


def test_agent_does_not_enforce_result_tool_usage(tmp_path: Path) -> None:
    """End a text-only agent run without inferring policy from result tool names."""

    provider = ProviderStreamMock(
        [final_text_stream("resp_1", "The temperature is 21C.")]
    )

    events = collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        provider=provider,
        tools=_result_tools(),
    )

    assert provider.await_count == 1
    assert not any(isinstance(event, ResultFollowUpEvent) for event in events)
    _agent_end_event(events)


def test_runtime_maps_fail_tool_to_failed_outcome() -> None:
    """Map a terminating fail tool result into the runtime's failed outcome."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_1",
                call_id="call_1",
                tool_name="fail",
                arguments={"reason": "The city is ambiguous."},
            ),
        ]
    )

    harness, configured_provider = _harness(provider)

    async def _run() -> Failed | None:
        """Run one result prompt and return its outcome when failed."""

        run = await harness.prompt(
            "Weather?", provider=configured_provider, result=WeatherReport
        )
        outcome = await run.wait()
        return outcome if isinstance(outcome, Failed) else None

    outcome = asyncio.run(_run())

    assert provider.await_count == 1
    assert outcome == Failed(cause=AgentFailure(reason="The city is ambiguous."))


def test_agent_retries_complete_after_validation_error(tmp_path: Path) -> None:
    """Route result validation errors back to the model for correction."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream("resp_1", "call_1", {"city": "Munich"}),
            _complete_call_stream(
                "resp_2", "call_2", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    events = collect_run_events(
        [UserMessage(content="Weather in Munich?")],
        provider=provider,
        tools=_result_tools(),
    )

    assert provider.await_count == 2
    retry_history = provider.history(1)
    error_result = retry_history[-1]
    assert isinstance(error_result, ToolResultTurn)
    assert error_result.is_error
    executions = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert not executions[0].outcome.terminate
    assert executions[1].outcome.terminate
    _agent_end_event(events)


def test_runtime_nudges_text_only_agent_run_toward_result() -> None:
    """Start another agent run with a persisted nudge after a text-only ending."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "The temperature is 21C."),
            _complete_call_stream(
                "resp_2", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )

    store = SQLiteStore(in_memory=True)
    harness, configured_provider = _harness(provider, store=store, session_id="nudged")

    async def _run() -> tuple[list[AgentEvent], Completed | None]:
        """Collect the complete runtime event stream and typed outcome."""

        run = await harness.prompt(
            "Weather in Munich?",
            provider=configured_provider,
            result=WeatherReport,
        )
        result = await run.wait()
        events = [event async for event in run.events()]
        return events, result if isinstance(result, Completed) else None

    events, outcome = asyncio.run(_run())

    follow_ups = [e for e in events if isinstance(e, ResultFollowUpEvent)]
    assert sum(isinstance(event, AgentStartEvent) for event in events) == 2
    assert sum(isinstance(event, AgentEndEvent) for event in events) == 2
    assert len(follow_ups) == 1
    assert follow_ups[0].message.content == RESULT_FOLLOW_UP
    nudged_history = provider.history(1)
    assert nudged_history[-1] == UserMessage(content=RESULT_FOLLOW_UP)
    assert UserMessage(content=RESULT_FOLLOW_UP) in tuple(
        item.item for item in store.get_history("nudged")
    )
    assert outcome is not None
    assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)


def test_runtime_fails_after_follow_up_cap() -> None:
    """Give up with a failure outcome when runtime nudges never produce a result."""

    streams = [
        final_text_stream(f"resp_{index}", "Still thinking.")
        for index in range(MAX_RESULT_FOLLOW_UPS + 1)
    ]
    provider = ProviderStreamMock(streams)

    harness, configured_provider = _harness(provider)

    async def _run() -> Failed | None:
        """Run until the output-contract follow-up limit is exhausted."""

        run = await harness.prompt(
            "Weather?", provider=configured_provider, result=WeatherReport
        )
        outcome = await run.wait()
        return outcome if isinstance(outcome, Failed) else None

    outcome = asyncio.run(_run())

    assert provider.await_count == MAX_RESULT_FOLLOW_UPS + 1
    assert outcome == Failed(cause=AgentFailure(reason=NO_RESULT_REASON))


def test_runtime_without_contract_completes_with_text() -> None:
    """Wrap a plain runtime prompt's text ending in a completed outcome."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "The temperature is 21C."),
        ]
    )

    harness, configured_provider = _harness(provider)

    async def _run() -> Completed | None:
        """Run one plain prompt and return its completed outcome."""

        run = await harness.prompt("Weather in Munich?", provider=configured_provider)
        outcome = await run.wait()
        return outcome if isinstance(outcome, Completed) else None

    outcome = asyncio.run(_run())

    assert outcome == Completed(value="The temperature is 21C.")


def test_runtime_fails_when_nudge_attempt_hits_stream_error() -> None:
    """Propagate a follow-up attempt's stream error, keeping stable history."""

    provider = ProviderStreamMock(
        [
            final_text_stream("resp_1", "Still thinking."),
            error_stream("resp_2", "boom"),
        ]
    )

    store = SQLiteStore(in_memory=True)
    harness, configured_provider = _harness(
        provider, store=store, session_id="nudged-error"
    )

    async def _run() -> None:
        """Fail the result prompt on its nudged second attempt."""

        run = await harness.prompt(
            "Weather?",
            provider=configured_provider,
            result=WeatherReport,
        )
        outcome = await run.wait()
        assert isinstance(outcome, Failed)
        assert isinstance(outcome.cause, ExecutionFailure)
        assert outcome.cause.message == "boom"

    asyncio.run(_run())

    assert provider.await_count == 2
    history = [item.item for item in store.get_history("nudged-error")]
    assert len(history) == 3
    assert history[0] == UserMessage(content="Weather?")
    first_attempt = history[1]
    assert isinstance(first_attempt, AssistantTurn)
    assert first_attempt.status == "completed"
    assert history[2] == UserMessage(content=RESULT_FOLLOW_UP)


def test_agent_finishes_tool_batch_after_terminating_result(tmp_path: Path) -> None:
    """Execute sibling tools before a terminating result ends the agent loop."""

    provider = ProviderStreamMock(
        [
            [
                stream_start("resp_1"),
                stream_done(
                    "resp_1",
                    stop_reason="tool_use",
                    blocks=[
                        tool_call_block(
                            call_id="call_1",
                            name="complete",
                            arguments={"city": "Munich", "temp_c": 21.0},
                        ),
                        tool_call_block(
                            call_id="call_2",
                            name="get_weather",
                            arguments={"city": "Berlin"},
                        ),
                    ],
                ),
            ],
        ]
    )

    events = collect_run_events(
        [UserMessage(content="Weather?")],
        provider=provider,
        tools=[*_result_tools(), city_tool("get_weather", "Get weather.", _weather)],
    )

    executions = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(executions) == 2
    assert executions[0].outcome.terminate
    assert not executions[1].outcome.tool_result_turn.is_error
    assert not executions[1].outcome.terminate
    assert provider.await_count == 1


def test_runtime_keeps_terminal_text_separate_from_result_value() -> None:
    """Expose terminal assistant text on the run without duplicating it in outcome."""

    provider = ProviderStreamMock(
        [
            [
                stream_start("resp_1"),
                stream_done(
                    "resp_1",
                    stop_reason="tool_use",
                    blocks=[
                        TextBlock(text="Recording the result."),
                        tool_call_block(
                            call_id="call_1",
                            name="complete",
                            arguments={"city": "Munich", "temp_c": 21.0},
                        ),
                    ],
                ),
            ],
        ]
    )

    harness, configured_provider = _harness(provider)

    async def _run() -> tuple[Completed | None, str | None]:
        """Run one result prompt and return its outcome and assistant text."""

        run = await harness.prompt(
            "Weather?", provider=configured_provider, result=WeatherReport
        )
        result = await run.wait()
        outcome = result if isinstance(result, Completed) else None
        assistant_turn = next(
            (
                item
                for item in reversed(harness.session.get_history())
                if isinstance(item, AssistantTurn)
            ),
            None,
        )
        assert assistant_turn is not None
        output_text = "".join(
            block.text
            for block in assistant_turn.blocks
            if isinstance(block, TextBlock)
        )
        return outcome, output_text

    outcome, output_text = asyncio.run(_run())

    assert outcome is not None
    assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)
    assert output_text == "Recording the result."


def test_session_prompt_composes_result_tools_and_contract() -> None:
    """Add result tools and contract for one prompt when a schema is set."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
        ]
    )
    store = SQLiteStore(in_memory=True)
    harness, configured_provider = _harness(
        provider, store=store, session_id="result-session"
    )

    async def _run() -> None:
        run = await harness.prompt(
            "Weather in Munich?",
            provider=configured_provider,
            result=WeatherReport,
        )
        outcome = await run.wait()
        assert isinstance(outcome, Completed)
        assert outcome.value == WeatherReport(city="Munich", temp_c=21.0)

    asyncio.run(_run())

    tools = provider.tools(0)
    assert tools is not None
    assert {tool.name for tool in tools} == {"complete", "fail"}
    instructions = provider.mock.await_args_list[0].kwargs["instructions"]
    assert RESULT_CONTRACT in instructions


def test_session_mixes_contract_and_plain_prompts() -> None:
    """Run contract and plain prompts back to back on one session."""

    provider = ProviderStreamMock(
        [
            _complete_call_stream(
                "resp_1", "call_1", {"city": "Munich", "temp_c": 21.0}
            ),
            final_text_stream("resp_2", "You asked about Munich."),
        ]
    )
    harness, configured_provider = _harness(provider, session_id="mixed-session")

    async def _run() -> None:
        contract_run = await harness.prompt(
            "Weather in Munich?",
            provider=configured_provider,
            result=WeatherReport,
        )
        contract_outcome = await contract_run.wait()
        assert isinstance(contract_outcome, Completed)
        assert contract_outcome.value == WeatherReport(city="Munich", temp_c=21.0)

        plain_run = await harness.prompt(
            "Which city did I ask about?", provider=configured_provider
        )
        assert await plain_run.wait() == Completed(value="You asked about Munich.")

    asyncio.run(_run())

    contract_tools = provider.tools(0)
    assert contract_tools is not None
    assert {tool.name for tool in contract_tools} == {"complete", "fail"}
    plain_tools = provider.tools(1)
    assert plain_tools == ()
    plain_instructions = provider.mock.await_args_list[1].kwargs["instructions"]
    assert RESULT_CONTRACT not in plain_instructions


def test_runtime_rejects_reserved_tool_names() -> None:
    """Refuse caller tools named after the reserved result tools."""

    with pytest.raises(ValueError, match="reserved"):
        AgentHarness(
            session=SessionRepository(SQLiteStore(in_memory=True)).create(),
            tools=[city_tool("complete", "Not the real complete.", _weather)],
            cwd=Path("."),
        )


def _harness(
    provider: ProviderStreamMock,
    *,
    store: SQLiteStore | None = None,
    session_id: str | None = None,
) -> tuple[AgentHarness, Provider]:
    """Build a session-bound harness for result-contract tests."""

    active_store = store if store is not None else SQLiteStore(in_memory=True)
    session = SessionRepository(active_store).create(session_id=session_id)
    harness = AgentHarness(session=session, auto_mode=False, cwd=Path("."))
    return harness, provider

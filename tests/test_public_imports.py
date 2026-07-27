"""Tests for the documented public import surface."""

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator, Sequence
from typing import get_args

from tile import (
    Aborted,
    AgentFailure,
    AgentRuntime,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    FailureCause,
    RunHandle,
    RunReport,
    RunRecord,
    Session,
    SessionNotFoundError,
    SQLiteStore,
    Store,
    StorePersistenceError,
    TurnFailedError,
)
from tile.events import AgentEvent, MessageEndEvent, RunEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.types import (
    AsyncEventStream,
    ConversationItem,
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamStartEvent,
    TextBlock,
    ToolDefinition,
    ToolError,
    ToolInput,
    ToolInputValidationFailure,
    ToolInvocationFailure,
    ToolResult,
)


def test_documented_public_imports_run_fake_prompt() -> None:
    """Run one prompt using only documented public imports."""

    store: Store = SQLiteStore(in_memory=True)
    runtime = AgentRuntime(
        stream_fn=_fake_stream_fn(),
        model="gpt-5.4",
        store=store,
        tools=[_fake_tool_definition()],
        cwd=Path("."),
    )
    session: Session = runtime.session(session_id="public-imports")

    events = asyncio.run(_collect_prompt_events(session))

    assert isinstance(events[-1], RunEndEvent)
    assert any(isinstance(event, MessageEndEvent) for event in events)
    assert len(session.history) == 2
    run_records = runtime.runs_for(session.id)
    assert len(run_records) == 1
    assert isinstance(run_records[0], RunRecord)
    assert run_records[0].status == "completed"
    assert issubclass(SessionNotFoundError, KeyError)
    assert issubclass(TurnFailedError, RuntimeError)
    assert ExecutionFailure.model_fields["origin"]
    assert get_args(ExecutionFailureOrigin) == ("turn", "execution")
    assert get_args(FailureCause) == (AgentFailure, ExecutionFailure)
    assert RunReport.__name__ == "RunReport"
    assert issubclass(StorePersistenceError, RuntimeError)
    assert Failed.model_fields["cause"]
    assert Aborted().type == "aborted"
    assert ToolInputValidationFailure.model_fields["issues"]
    assert ToolInvocationFailure.model_fields["exception_type"]
    assert issubclass(ToolError, RuntimeError)
    assert callable(create_stream_api)


def _fake_stream_fn() -> StreamFn:
    """Build a fake provider stream from public provider-neutral contracts."""

    return _FakeStreamFn()


class _FakeStreamFn:
    """Fake provider stream function carrying its declared provider identity."""

    provider = "fake"

    async def __call__(
        self,
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return a deterministic assistant response."""

        _ = instructions
        assert len(history) == 1
        assert model == "gpt-5.4"
        assert tools is not None
        return _stream_events(_assistant_response())


def _assistant_response() -> tuple[ProviderStreamEvent, ...]:
    """Build a minimal successful provider event stream."""

    source = ProviderSource(provider="fake", model="gpt-5.4")
    return (
        StreamStartEvent(source=source, response_id="resp_public"),
        StreamDoneEvent(
            source=source,
            response_id="resp_public",
            stop_reason="stop",
            blocks=[TextBlock(text="public import response")],
        ),
    )


async def _stream_events(
    events: Sequence[ProviderStreamEvent],
) -> AsyncGenerator[ProviderStreamEvent, None]:
    """Yield fake provider events."""

    for event in events:
        yield event


async def _collect_prompt_events(session: Session) -> list[AgentEvent]:
    """Collect all runtime events for one prompt."""

    run: RunHandle = await session.prompt("hello")
    return [event async for event in run.events()]


class _FakeToolInput(ToolInput):
    """Strict empty public input contract for the fake tool."""


async def _fake_tool(params: _FakeToolInput) -> ToolResult:
    """Return a deterministic tool result for import smoke coverage."""

    _ = params
    return ToolResult.text("ok")


def _fake_tool_definition() -> ToolDefinition:
    """Build a tool definition from the public tool contract."""

    return ToolDefinition(
        name="fake_tool",
        description="Return a deterministic value.",
        input_model=_FakeToolInput,
        fn=_fake_tool,
    )

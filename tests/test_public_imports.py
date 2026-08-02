"""Tests for the documented public import surface."""

from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import get_args

import tile
from tile import (
    Aborted,
    AgentFailure,
    AgentHarness,
    ExecutionFailure,
    ExecutionFailureOrigin,
    Failed,
    FailureCause,
    Provider,
    RunAlreadyEndedError,
    RunHandle,
    RunRecord,
    Session,
    SessionNotFoundError,
    SessionRepository,
    SQLiteStore,
    Store,
    StorePersistenceError,
    TurnFailedError,
)
from tile.events import AgentEvent, MessageEndEvent, RunEndEvent
from tile.providers.openai import OpenAIProvider
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


async def test_documented_public_imports_run_fake_prompt() -> None:
    """Run one prompt using only documented public imports."""

    store: Store = SQLiteStore(in_memory=True)
    session: Session = SessionRepository(store).create(session_id="public-imports")
    harness = AgentHarness(
        session=session,
        tools=[_fake_tool_definition()],
        cwd=Path("."),
    )
    provider = _FakeProvider()

    events = await _collect_prompt_events(harness, provider)

    assert isinstance(events[-1], RunEndEvent)
    assert any(isinstance(event, MessageEndEvent) for event in events)
    assert len(session.get_history()) == 2
    run_records = session.get_runs()
    assert len(run_records) == 1
    assert isinstance(run_records[0], RunRecord)
    assert run_records[0].status == "completed"
    assert run_records[0].provider == provider.name
    assert issubclass(SessionNotFoundError, KeyError)
    assert issubclass(TurnFailedError, RuntimeError)
    assert ExecutionFailure.model_fields["origin"]
    assert get_args(ExecutionFailureOrigin) == ("turn", "execution")
    assert get_args(FailureCause) == (AgentFailure, ExecutionFailure)
    assert issubclass(RunAlreadyEndedError, RuntimeError)
    assert issubclass(StorePersistenceError, RuntimeError)
    assert Failed.model_fields["cause"]
    assert Aborted().type == "aborted"
    assert ToolInputValidationFailure.model_fields["issues"]
    assert ToolInvocationFailure.model_fields["exception_type"]
    assert issubclass(ToolError, RuntimeError)
    assert issubclass(OpenAIProvider, Provider)


def test_removed_runtime_concepts_are_not_public_exports() -> None:
    """Keep replaced runtime concepts outside the top-level package facade."""

    for name in ("AgentRuntime", "RunReport", "StartedRun"):
        assert not hasattr(tile, name)


class _FakeProvider(Provider):
    """Fake configured provider built from public neutral contracts."""

    def __init__(self) -> None:
        """Configure the fake model."""

        super().__init__(model="gpt-5.4")

    @property
    def name(self) -> str:
        """Return the fake provider identity."""

        return "fake"

    async def stream(
        self,
        history: Sequence[ConversationItem],
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncEventStream:
        """Return a deterministic assistant response."""

        _ = instructions
        assert len(history) == 1
        assert self.model == "gpt-5.4"
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


async def _collect_prompt_events(
    harness: AgentHarness,
    provider: Provider,
) -> list[AgentEvent]:
    """Collect all harness events for one prompt."""

    run: RunHandle = await harness.prompt("hello", provider=provider)
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

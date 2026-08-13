"""Tests for extension assembly, lifecycle hooks, and observers."""

import asyncio
import logging
from pathlib import Path
from typing import cast

import pytest

from tests.support.agent_streams import (
    GatedProviderStreamMock,
    ProviderStreamMock,
    final_text_stream,
    tool_call_block,
)
from tile import AgentHarness, Completed, SessionRepository
from tile.events import AgentEvent
from tile.extensions import (
    BeforeRunContext,
    BeforeRunHook,
    BeforeRunResult,
    EventLogger,
    ExtensionRegistry,
    NonInteractive,
    RunEventStream,
)
from tile.store import SQLiteStore
from tile.types import (
    AssistantTurn,
    ToolResultTurn,
    ToolTextContent,
    UserMessage,
)


class _ChainedHooks:
    """Contribute ordered hooks and record the contexts they observe."""

    def __init__(self) -> None:
        """Create an unregistered extension."""

        self.registrations = 0
        self.seen: list[BeforeRunContext] = []

    def register(self, registry: ExtensionRegistry) -> None:
        """Register two hooks once during harness construction."""

        self.registrations += 1
        registry.before_run(self._first)
        registry.before_run(self._second)

    async def _first(self, context: BeforeRunContext) -> BeforeRunResult:
        """Replace instructions and append one run input message."""

        self.seen.append(context)
        return BeforeRunResult(
            system_prompt="first hook",
            additional_messages=(UserMessage(content="extension context"),),
        )

    async def _second(self, context: BeforeRunContext) -> BeforeRunResult:
        """Observe the first result and replace its instructions."""

        self.seen.append(context)
        return BeforeRunResult(system_prompt="second hook")


async def test_before_run_hooks_chain_and_commit_their_final_input(
    store: SQLiteStore,
) -> None:
    """Apply hooks in order and commit their final context at run end."""

    extension = _ChainedHooks()
    release = asyncio.Event()
    provider = GatedProviderStreamMock([release])
    session = SessionRepository(store).create(session_id="hook-chain")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="core",
        extensions=(extension,),
    )

    run = await harness.prompt("caller prompt", provider=provider)
    await provider.wait_for_calls()

    assert extension.registrations == 1
    assert extension.seen[0].system_prompt.startswith("core")
    assert extension.seen[1].system_prompt == "first hook"
    assert extension.seen[1].messages == (
        UserMessage(content="caller prompt"),
        UserMessage(content="extension context"),
    )
    record = store.get_run(session.id, run.id)
    assert record.prompt == "caller prompt"
    assert session.get_history() == ()
    assert provider.instructions() == "second hook"
    assert provider.history(0) == extension.seen[1].messages

    release.set()
    assert await run.wait() == Completed(value="answer 0")
    assert session.get_history()[:2] == extension.seen[1].messages


class _FailingHook:
    """Reject a run before it reaches persistence."""

    def register(self, registry: ExtensionRegistry) -> None:
        """Register the failing handler."""

        registry.before_run(self.before_run)

    async def before_run(self, context: BeforeRunContext) -> None:
        """Raise a deterministic admission failure."""

        _ = context
        raise LookupError("missing extension configuration")


async def test_before_run_failure_rejects_admission(
    store: SQLiteStore,
) -> None:
    """Propagate hook failures without creating a run or invoking a provider."""

    provider = ProviderStreamMock([final_text_stream("response-1", "unused")])
    session = SessionRepository(store).create(session_id="hook-failure")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(_FailingHook(),),
    )

    with pytest.raises(LookupError, match="missing extension configuration"):
        await harness.prompt("hello", provider=provider)

    assert session.get_runs() == ()
    assert session.get_history() == ()
    assert provider.await_count == 0


class _InvalidResultHook:
    """Return a value that violates the typed hook contract."""

    def register(self, registry: ExtensionRegistry) -> None:
        """Simulate registration from an untyped extension."""

        registry.before_run(cast(BeforeRunHook, self.before_run))

    async def before_run(self, context: BeforeRunContext) -> str:
        """Return an invalid hook result."""

        _ = context
        return "invalid"


async def test_before_run_rejects_an_invalid_result_type(
    store: SQLiteStore,
) -> None:
    """Reject runtime contract violations before run admission."""

    provider = ProviderStreamMock([final_text_stream("response-1", "unused")])
    session = SessionRepository(store).create(session_id="invalid-result-type")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(_InvalidResultHook(),),
    )

    with pytest.raises(TypeError, match="BeforeRunResult or None"):
        await harness.prompt("hello", provider=provider)

    assert session.get_runs() == ()
    assert provider.await_count == 0


class _ToolExchangeHook:
    """Add a tool exchange to the initial run conversation."""

    def __init__(self, *, include_result: bool) -> None:
        """Choose whether the injected tool call is complete."""

        self._include_result = include_result

    def register(self, registry: ExtensionRegistry) -> None:
        """Register the tool-exchange hook."""

        registry.before_run(self.before_run)

    async def before_run(self, context: BeforeRunContext) -> BeforeRunResult:
        """Return one tool call and, when requested, its result."""

        _ = context
        tool_call = AssistantTurn(
            blocks=[
                tool_call_block(
                    call_id="extension-call",
                    name="extension_tool",
                    arguments={},
                )
            ]
        )
        if not self._include_result:
            return BeforeRunResult(additional_messages=(tool_call,))
        tool_result = ToolResultTurn(
            call_id="extension-call",
            tool_name="extension_tool",
            content=[ToolTextContent(text="injected result")],
        )
        return BeforeRunResult(additional_messages=(tool_call, tool_result))


async def test_before_run_rejects_an_unanswered_tool_call(
    store: SQLiteStore,
) -> None:
    """Validate each hook result before admitting its messages."""

    provider = ProviderStreamMock([final_text_stream("response-1", "unused")])
    session = SessionRepository(store).create(session_id="invalid-hook-result")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(_ToolExchangeHook(include_result=False),),
    )

    with pytest.raises(ValueError, match="tool calls require results"):
        await harness.prompt("hello", provider=provider)

    assert session.get_runs() == ()
    assert session.get_history() == ()
    assert provider.await_count == 0


async def test_before_run_accepts_a_complete_tool_exchange(
    store: SQLiteStore,
) -> None:
    """Admit matching tool calls and results returned by one hook."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    session = SessionRepository(store).create(session_id="valid-hook-result")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(_ToolExchangeHook(include_result=True),),
    )

    run = await harness.prompt("hello", provider=provider)

    assert await run.wait() == Completed(value="done")
    assert len(provider.history(0)) == 3


async def test_non_interactive_is_an_explicit_before_run_extension(
    store: SQLiteStore,
) -> None:
    """Prepend the non-interactive execution policy for each run."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    session = SessionRepository(store).create(session_id="non-interactive")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="core",
        extensions=(NonInteractive(),),
    )

    run = await harness.prompt("hello", provider=provider)
    assert await run.wait() == Completed(value="done")

    assert provider.instructions().startswith(
        "You are operating inside Tile, a headless agent runtime."
    )
    assert "\n\ncore" in provider.instructions()


class _ObservingExtension:
    """Exercise independent, failure-isolated run event streams."""

    def __init__(self) -> None:
        """Create empty observation state."""

        self.events: list[AgentEvent] = []
        self.identities: list[tuple[str, str]] = []
        self.cancelled = asyncio.Event()
        self.completed = asyncio.Event()

    def register(self, registry: ExtensionRegistry) -> None:
        """Register failure, mutation, and collection stream consumers."""

        registry.observe(self._raise)
        registry.observe(self._cancel)
        registry.observe(self._mutate_copy)
        registry.observe(self._collect)

    async def _raise(self, stream: RunEventStream) -> None:
        """Prove one observer cannot interrupt execution or another stream."""

        async for event in stream:
            raise LookupError(event.type)

    async def _cancel(self, stream: RunEventStream) -> None:
        """Prove observer cancellation remains local to its own task."""

        async for _ in stream:
            self.cancelled.set()
            raise asyncio.CancelledError

    async def _mutate_copy(self, stream: RunEventStream) -> None:
        """Try to alter every event yielded to one observer."""

        async for event in stream:
            event.type = "mutated"

    async def _collect(self, stream: RunEventStream) -> None:
        """Capture one independent stream and its stable run identity."""

        self.identities.append((stream.session_id, stream.run_id))
        self.events.extend([event async for event in stream])
        self.completed.set()


async def test_run_observers_receive_identity_without_affecting_execution(
    store: SQLiteStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Give each observer a complete copy-safe stream and isolate failures."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    session = SessionRepository(store).create(session_id="observed-session")
    extension = _ObservingExtension()
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(extension,),
    )

    with caplog.at_level(logging.ERROR, logger="tile.extensions.run_observers"):
        run = await harness.prompt("hello", provider=provider)
        assert await run.wait() == Completed(value="done")
        await asyncio.wait_for(extension.cancelled.wait(), timeout=1)
        await asyncio.wait_for(extension.completed.wait(), timeout=1)
        caller_events = [event async for event in run.events()]

    assert extension.events[0].type == "run_start"
    assert extension.events[-1].type == "run_end"
    assert all(event.type != "mutated" for event in extension.events)
    assert all(event.type != "mutated" for event in caller_events)
    assert extension.identities == [(session.id, run.id)]
    assert "Tile run observer failed" in caplog.text


class _DelayedObserver:
    """Delay consumption until after its observed run has finalized."""

    def __init__(self) -> None:
        """Create explicit release and completion signals."""

        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.events: list[AgentEvent] = []

    def register(self, registry: ExtensionRegistry) -> None:
        """Register the delayed stream consumer."""

        registry.observe(self.observe)

    async def observe(self, stream: RunEventStream) -> None:
        """Drain the buffered stream only after the test releases it."""

        self.started.set()
        await self.release.wait()
        self.events.extend([event async for event in stream])
        self.completed.set()


async def test_run_observer_can_drain_events_after_wait_returns(
    store: SQLiteStore,
) -> None:
    """Retain a complete observer stream beyond run finalization."""

    observer = _DelayedObserver()
    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    session = SessionRepository(store).create(session_id="delayed-observer")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(observer,),
    )

    run = await harness.prompt("hello", provider=provider)
    try:
        await asyncio.wait_for(observer.started.wait(), timeout=1)
        assert await asyncio.wait_for(run.wait(), timeout=1) == Completed(value="done")
        assert not observer.completed.is_set()
    finally:
        observer.release.set()
    await asyncio.wait_for(observer.completed.wait(), timeout=1)

    assert observer.events[0].type == "run_start"
    assert observer.events[-1].type == "run_end"


async def test_event_logger_logs_every_observed_run_event(
    store: SQLiteStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose logging as an explicit built-in observer extension."""

    logger = logging.getLogger("tests.tile.events")
    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    session = SessionRepository(store).create(session_id="logged-session")
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
        extensions=(EventLogger(logger),),
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        run = await harness.prompt("hello", provider=provider)
        assert await run.wait() == Completed(value="done")
        await asyncio.wait_for(
            _wait_for_log(caplog, "event={'type': 'run_end'"),
            timeout=1,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("event={'type': 'run_start'}" in message for message in messages)
    assert any("event={'type': 'run_end'" in message for message in messages)
    assert all(f"session_id={session.id}" in message for message in messages)
    assert all(f"run_id={run.id}" in message for message in messages)


def test_extension_registry_does_not_register_tools() -> None:
    """Keep model-callable tools on the explicit harness API."""

    assert not hasattr(ExtensionRegistry(), "add_tools")


async def _wait_for_log(
    caplog: pytest.LogCaptureFixture,
    expected: str,
) -> None:
    """Yield until one asynchronously observed log message arrives."""

    while not any(expected in record.getMessage() for record in caplog.records):
        await asyncio.sleep(0)

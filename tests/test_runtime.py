"""End-to-end tests for the persistence-first runtime."""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from tile import (
    Aborted,
    ActiveRunError,
    AgentRuntime,
    Completed,
    ExecutionFailure,
    Failed,
    PersistenceFailure,
    RunHandle,
    RunPersistenceError,
    RunOutcome,
    RunRecord,
    SessionNotFoundError,
    SQLiteStore,
)
from tile.events import AgentEvent, RunEndEvent
from tile.types import (
    AsyncEventStream,
    ConversationItem,
    ToolDefinition,
    ToolFunction,
    ToolInput,
    ToolResult,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import ProviderStreamEvent
from tests.support.agent_streams import (
    GatedProviderStreamMock,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    tool_call_stream,
)
from tests.support.async_streams import async_stream


class _NoInput(ToolInput):
    """Strict empty input for deterministic runtime tools."""


def test_runtime_creates_lists_and_gets_persistent_sessions() -> None:
    """Use the Store as the authoritative session registry."""

    runtime, _, _ = _runtime([])
    generated = runtime.session(name="Generated")
    explicit = runtime.session(session_id="known", name="Known")
    repeated = runtime.session(session_id="known", name="Ignored")

    assert generated.id != explicit.id
    assert repeated.name == "Known"
    assert [session.id for session in runtime.sessions] == [generated.id, "known"]
    assert runtime.get_session("known").name == "Known"
    with pytest.raises(SessionNotFoundError, match="missing"):
        runtime.get_session("missing")


def test_runtime_commits_a_complete_turn_only_at_finalization() -> None:
    """Keep prompt and assistant output provisional while the run is active."""

    async def _run() -> None:
        release = asyncio.Event()
        provider = GatedProviderStreamMock([release])
        store = SQLiteStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        session = runtime.session(session_id="atomic-history")
        handle = await session.prompt("hello")
        await _wait_for_provider(provider)

        assert session.history == ()
        assert handle.conversation_items == (handle.conversation_items[0],)
        prompt = handle.conversation_items[0]
        assert isinstance(prompt, UserMessage)
        assert prompt.content == "hello"
        assert store.get_run(handle.id).status == "running"
        assert store.get_run(handle.id).prompt == "hello"

        release.set()
        assert await handle.wait() == "completed"
        assert len(session.history) == 2
        assert store.get_run(handle.id).outcome == Completed(value="answer 0")
        store.close()

    asyncio.run(_run())


def test_runtime_persists_running_record_before_provider_execution() -> None:
    """Make the submitted prompt durable before invoking the provider."""

    async def _run() -> None:
        store = SQLiteStore(in_memory=True)
        observed_statuses: list[str] = []

        class _ObservingProvider:
            """Read the store at the provider boundary."""

            provider = "test"

            async def __call__(
                self,
                history: Sequence[ConversationItem],
                model: str,
                *,
                instructions: str,
                tools: Sequence[ToolDefinition] | None,
            ) -> AsyncEventStream:
                """Capture the running record before returning a response."""

                _ = history, model, instructions, tools
                observed_statuses.extend(
                    run.status for run in store.list_runs("session-1")
                )
                return async_stream(final_text_stream("response-1", "done"))

        runtime = AgentRuntime(
            stream_fn=_ObservingProvider(),
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        run = await runtime.session(session_id="session-1").prompt("hello")
        assert await run.wait() == "completed"
        assert observed_statuses == ["running"]
        store.close()

    asyncio.run(_run())


def test_runtime_failure_commits_the_valid_completed_prefix() -> None:
    """Persist replayable items while excluding an errored assistant turn."""

    runtime, store, _ = _runtime([error_stream("response-1", "provider unavailable")])
    session = runtime.session(session_id="failed")

    async def _run() -> RunHandle:
        handle = await session.prompt("hello")
        assert await handle.wait() == "failed"
        return handle

    handle = asyncio.run(_run())

    assert handle.outcome == Failed(
        cause=ExecutionFailure(
            origin="turn",
            exception_type="TurnFailedError",
            message="provider unavailable",
        )
    )
    assert session.history == (session.history[0],)
    prompt = session.history[0]
    assert isinstance(prompt, UserMessage)
    assert prompt.content == "hello"
    store.close()


def test_abort_commits_a_healed_replayable_prefix() -> None:
    """Heal an interrupted tool call before atomically committing an abort."""

    async def _blocked_tool(params: _NoInput) -> ToolResult:
        """Block until task cancellation interrupts execution."""

        _ = params
        await asyncio.Event().wait()
        return ToolResult.text("unreachable")

    tool = _tool("blocked", _blocked_tool)
    runtime, store, provider = _runtime(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="blocked",
                arguments={},
            )
        ],
        tools=[tool],
    )
    session = runtime.session(session_id="abort")

    async def _run() -> RunHandle:
        handle = await session.prompt("start")
        await _wait_for_provider(provider)
        for _ in range(20):
            if len(handle.conversation_items) >= 2:
                break
            await asyncio.sleep(0)
        handle.abort()
        assert await handle.wait() == "aborted"
        return handle

    handle = asyncio.run(_run())
    history = session.history

    assert handle.outcome == Aborted(reason="cancelled")
    assert len(history) == 3
    healed = history[-1]
    assert isinstance(healed, ToolResultTurn)
    assert healed.is_error
    store.close()


def test_overlapping_prompt_is_rejected_by_the_store() -> None:
    """Enforce one running run without a process-local session lock."""

    async def _run() -> None:
        release = asyncio.Event()
        provider = GatedProviderStreamMock([release])
        store = SQLiteStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        session = runtime.session(session_id="busy")
        first = await session.prompt("first")
        await _wait_for_provider(provider)

        with pytest.raises(ActiveRunError, match="busy"):
            await session.prompt("second")

        first.abort()
        assert await first.wait() == "aborted"
        store.close()

    asyncio.run(_run())


def test_replace_active_fences_old_history_and_runs_the_successor() -> None:
    """Cancel a local predecessor after atomic replacement and commit only the new turn."""

    async def _run() -> None:
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        provider = GatedProviderStreamMock([first_release, second_release])
        store = SQLiteStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        session = runtime.session(session_id="replace")
        first = await session.prompt("first")
        await _wait_for_provider(provider)
        second = await session.prompt("second", replace_active=True)
        await _wait_for_provider(provider, expected=2)

        assert await first.wait() == "aborted"
        assert first.outcome == Aborted(reason="replaced")
        second_release.set()
        assert await second.wait() == "completed"
        assert [
            item.content for item in session.history if isinstance(item, UserMessage)
        ] == ["second"]
        assert store.get_run(first.id).outcome == Aborted(reason="replaced")
        store.close()

    asyncio.run(_run())


def test_replace_active_works_across_runtime_instances(tmp_path: Path) -> None:
    """Use database fencing when the predecessor belongs to another runtime."""

    async def _run() -> None:
        database_path = tmp_path / "shared.db"
        first_store = SQLiteStore(database_path)
        second_store = SQLiteStore(database_path)
        first_release = asyncio.Event()
        first_provider = GatedProviderStreamMock([first_release])
        second_provider = ProviderStreamMock(
            [final_text_stream("response-2", "replacement")]
        )
        first_runtime = AgentRuntime(
            stream_fn=first_provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=first_store,
        )
        second_runtime = AgentRuntime(
            stream_fn=second_provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=second_store,
        )
        first_session = first_runtime.session(session_id="shared")
        first = await first_session.prompt("first")
        await _wait_for_provider(first_provider)

        second_session = second_runtime.get_session("shared")
        second = await second_session.prompt("second", replace_active=True)
        assert await second.wait() == "completed"
        first_release.set()
        assert await first.wait() == "aborted"
        assert first.outcome == Aborted(reason="replaced")
        assert [
            item.content
            for item in second_session.history
            if isinstance(item, UserMessage)
        ] == ["second"]
        first_store.close()
        second_store.close()

    asyncio.run(_run())


def test_atomic_finalization_failure_is_visible_and_recoverable() -> None:
    """Raise to waiters, keep the run active, and permit explicit replacement."""

    async def _run() -> None:
        store = _FailingFinishStore(in_memory=True)
        provider = ProviderStreamMock(
            [
                final_text_stream("response-1", "lost"),
                final_text_stream("response-2", "recovered"),
            ]
        )
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        session = runtime.session(session_id="failure")
        first = await session.prompt("first")

        with pytest.raises(RunPersistenceError, match=first.id):
            await first.wait()

        events = [event async for event in first.events()]
        terminal = events[-1]
        assert isinstance(terminal, RunEndEvent)
        assert isinstance(terminal.outcome, Failed)
        assert isinstance(terminal.outcome.cause, PersistenceFailure)
        assert store.get_run(first.id).status == "running"
        assert session.history == ()

        store.fail_finishes = False
        recovery = await session.prompt("second", replace_active=True)
        assert await recovery.wait() == "completed"
        store.close()

    asyncio.run(_run())


def test_forked_session_inherits_flat_history_and_diverges() -> None:
    """Fork committed history without copying run ownership."""

    runtime, store, _ = _runtime(
        [
            final_text_stream("response-1", "source"),
            final_text_stream("response-2", "fork"),
        ]
    )
    source = runtime.session(session_id="source")

    async def _run() -> None:
        first = await source.prompt("first")
        assert await first.wait() == "completed"
        fork = source.fork(session_id="fork", name="Fork")
        assert fork.history == source.history
        assert runtime.runs_for("fork") == ()

        second = await fork.prompt("second")
        assert await second.wait() == "completed"
        assert len(fork.history) == 4
        assert len(source.history) == 2

    asyncio.run(_run())
    store.close()


def test_multiple_subscribers_replay_the_same_closed_log() -> None:
    """Keep event subscription independent from execution ownership."""

    runtime, store, _ = _runtime([final_text_stream("response-1", "done")])
    session = runtime.session(session_id="subscribers")

    async def _run() -> None:
        handle = await session.prompt("hello")
        first, second = await asyncio.gather(
            _collect_events(handle),
            _collect_events(handle),
        )
        assert first == second
        assert first[-1].type == "run_end"

    asyncio.run(_run())
    store.close()


def test_run_continues_after_a_subscriber_stops_consuming() -> None:
    """Keep execution task-owned when an event subscriber disconnects."""

    runtime, store, _ = _runtime([final_text_stream("response-1", "done")])
    session = runtime.session(session_id="early-stop")

    async def _run() -> None:
        handle = await session.prompt("hello")
        async for _ in handle.events():
            break

        assert await handle.wait() == "completed"
        assert len(session.history) == 2

    asyncio.run(_run())
    store.close()


def test_next_prompt_replays_committed_history_and_current_prompt() -> None:
    """Build later requests from typed committed history plus the new prompt."""

    runtime, store, provider = _runtime(
        [
            final_text_stream("response-1", "first answer"),
            final_text_stream("response-2", "second answer"),
        ]
    )
    session = runtime.session(session_id="multi-turn")

    async def _run() -> None:
        first = await session.prompt("first")
        assert await first.wait() == "completed"
        second = await session.prompt("second")
        assert await second.wait() == "completed"

    asyncio.run(_run())
    replayed = provider.history(1)
    assert len(replayed) == 3
    assert isinstance(replayed[0], UserMessage)
    assert replayed[0].content == "first"
    assert replayed[1].role == "assistant"
    assert isinstance(replayed[2], UserMessage)
    assert replayed[2].content == "second"
    store.close()


def test_run_handle_exposes_prompt_and_completed_output() -> None:
    """Expose the full provisional delta and latest completed assistant text."""

    runtime, store, _ = _runtime([final_text_stream("response-1", "done")])

    async def _run() -> None:
        handle = await runtime.session(session_id="output").prompt("hello")
        assert handle.output_text is None
        assert await handle.wait() == "completed"
        assert handle.output_text == "done"
        assert [item.role for item in handle.conversation_items] == [
            "user",
            "assistant",
        ]

    asyncio.run(_run())
    store.close()


def test_runtime_binds_cwd_and_rejects_model_visible_cwd(tmp_path: Path) -> None:
    """Inject cwd into tools while rejecting a conflicting input schema."""

    captured: list[Path] = []

    async def inspect_cwd(params: _NoInput, *, cwd: Path) -> ToolResult:
        """Capture the injected path."""

        _ = params
        captured.append(cwd)
        return ToolResult.text("ok")

    valid = _tool("inspect_cwd", inspect_cwd)
    runtime, store, _ = _runtime(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="inspect_cwd",
                arguments={},
            ),
            final_text_stream("response-2", "done"),
        ],
        tools=[valid],
        cwd=tmp_path,
    )

    async def _run() -> None:
        run = await runtime.session(session_id="cwd").prompt("inspect")
        assert await run.wait() == "completed"

    asyncio.run(_run())
    assert captured == [tmp_path.resolve()]
    store.close()


class _FailingFinishStore(SQLiteStore):
    """SQLite Store with a deterministic finalization failure."""

    fail_finishes: bool = True

    def finish_run(
        self,
        *,
        run_id: str,
        outcome: RunOutcome,
        history_delta: Sequence[ConversationItem],
        provider: str | None = None,
        model: str | None = None,
        ended_at: datetime | None = None,
    ) -> RunRecord:
        """Fail or delegate one atomic finish operation."""

        if self.fail_finishes:
            raise OSError("disk full")
        return super().finish_run(
            run_id=run_id,
            outcome=outcome,
            history_delta=history_delta,
            provider=provider,
            model=model,
            ended_at=ended_at,
        )


def _runtime(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    *,
    tools: Sequence[ToolDefinition] = (),
    cwd: Path = Path("."),
) -> tuple[AgentRuntime, SQLiteStore, ProviderStreamMock]:
    """Build a runtime with one in-memory SQLite Store."""

    provider = ProviderStreamMock(streams)
    store = SQLiteStore(in_memory=True)
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=cwd,
        store=store,
        tools=tools,
    )
    return runtime, store, provider


async def _wait_for_provider(
    provider: ProviderStreamMock,
    *,
    expected: int = 1,
) -> None:
    """Wait for a deterministic provider invocation."""

    for _ in range(30):
        if provider.await_count >= expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {expected} provider invocation(s).")


async def _collect_events(handle: RunHandle) -> list[AgentEvent]:
    """Collect one complete live run log."""

    return [event async for event in handle.events()]


def _tool(name: str, fn: ToolFunction) -> ToolDefinition:
    """Build a no-input test tool."""

    return ToolDefinition(
        name=name,
        description=f"Exercise {name}.",
        input_model=_NoInput,
        fn=fn,
    )

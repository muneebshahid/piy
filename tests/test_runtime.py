"""End-to-end tests for the persistence-first runtime."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from tile import (
    Aborted,
    ActiveRunError,
    AgentFailure,
    AgentRuntime,
    Completed,
    ExecutionFailure,
    Failed,
    HistoryItem,
    RunHandle,
    RunOutcome,
    RunRecord,
    SessionRecord,
    SessionNotFoundError,
    SQLiteStore,
    StorePersistenceError,
)
from tile.events import AgentEvent, RunEndEvent
from tile.types import (
    AssistantTurn,
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
    GatedQueuedProviderStreamMock,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    tool_call_stream,
)
from tests.support.async_streams import async_stream


class _NoInput(ToolInput):
    """Strict empty input for deterministic runtime tools."""


class _TextResult(BaseModel):
    """Minimal typed result for explicit agent-failure tests."""

    value: str


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
        assert store.get_run(handle.id).status == "running"
        assert store.get_run(handle.id).prompt == "hello"

        release.set()
        report = await handle.wait()
        assert report.status == "completed"
        prompt = report.history_delta[0]
        assert isinstance(prompt, UserMessage)
        assert prompt.content == "hello"
        assert len(session.history) == 2
        assert store.get_run(handle.id).outcome == Completed(value="answer 0")
        store.close()

    asyncio.run(_run())


def test_runtime_keeps_multi_attempt_history_provisional_until_finalization() -> None:
    """Keep assistant, tool, and result-follow-up items local while running."""

    async def _run() -> None:
        releases = (asyncio.Event(), asyncio.Event(), asyncio.Event())
        releases[0].set()
        releases[1].set()
        provider = GatedQueuedProviderStreamMock(
            [
                tool_call_stream(
                    response_id="response-1",
                    call_id="call-1",
                    tool_name="inspect",
                    arguments={},
                ),
                final_text_stream("response-2", "inspection complete"),
                tool_call_stream(
                    response_id="response-3",
                    call_id="call-2",
                    tool_name="complete",
                    arguments={"value": "done"},
                ),
            ],
            releases,
        )

        async def inspect(params: _NoInput) -> ToolResult:
            """Return one replayable non-terminal tool result."""

            _ = params
            return ToolResult.text("inspected")

        store = SQLiteStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
            tools=[_tool("inspect", inspect)],
        )
        session = runtime.session(session_id="provisional")
        handle = await session.prompt("inspect", result=_TextResult)

        await _wait_for_provider(provider, expected=3)
        assert session.history == ()
        assert [item.role for item in provider.history(2)] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
            "user",
        ]

        releases[2].set()
        report = await handle.wait()
        assert report.persisted
        assert [item.role for item in session.history] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
            "user",
            "assistant",
            "tool_result",
        ]
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
        assert (await run.wait()).status == "completed"
        assert observed_statuses == ["running"]
        store.close()

    asyncio.run(_run())


def test_runtime_bootstraps_from_start_run_history_snapshot() -> None:
    """Avoid a second fallible Store read after durable run acceptance."""

    store = _UnavailablePublicHistoryStore(in_memory=True)
    store.create_session(record=SessionRecord.create(session_id="session-1"))
    started = store.start_run(
        record=RunRecord.start(
            run_id="seed",
            session_id="session-1",
            prompt="first",
            model="gpt-5.4",
            provider="test",
        ),
    )
    store.finish_run(
        record=started.run.finish(outcome=Completed(value="first answer")),
        history_delta=[
            UserMessage(content="first"),
            AssistantTurn(response_id="response-1"),
        ],
    )
    provider = ProviderStreamMock([final_text_stream("response-2", "second answer")])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        store=store,
    )

    async def _run() -> None:
        handle = await runtime.get_session("session-1").prompt("second")
        assert (await handle.wait()).status == "completed"

    asyncio.run(_run())
    assert [item.role for item in provider.history(0)] == [
        "user",
        "assistant",
        "user",
    ]
    store.close()


def test_runtime_failure_commits_the_valid_completed_prefix() -> None:
    """Persist replayable items while excluding an errored assistant turn."""

    runtime, store, _ = _runtime([error_stream("response-1", "provider unavailable")])
    session = runtime.session(session_id="failed")

    async def _run() -> tuple[RunHandle, RunOutcome]:
        handle = await session.prompt("hello")
        report = await handle.wait()
        assert report.status == "failed"
        return handle, report.outcome

    handle, outcome = asyncio.run(_run())

    assert outcome == Failed(
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


def test_agent_failure_commits_its_complete_replayable_history() -> None:
    """Persist an agent-declared failure and every replayable item it produced."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="fail",
                arguments={"reason": "cannot deliver"},
            )
        ]
    )
    store = SQLiteStore(in_memory=True)
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        store=store,
    )
    session = runtime.session(session_id="agent-failure-history")

    async def _run() -> None:
        handle = await session.prompt("try", result=_TextResult)
        report = await handle.wait()

        expected = Failed(cause=AgentFailure(reason="cannot deliver"))
        assert report.outcome == expected
        assert report.persisted
        assert store.get_run(handle.id).outcome == expected
        assert [item.role for item in session.history] == [
            "user",
            "assistant",
            "tool_result",
        ]
        assert tuple(session.history) == report.history_delta

    asyncio.run(_run())
    store.close()


def test_abort_commits_a_healed_replayable_prefix() -> None:
    """Heal an interrupted tool call before atomically committing an abort."""

    tool_started = asyncio.Event()

    async def _blocked_tool(params: _NoInput) -> ToolResult:
        """Block until task cancellation interrupts execution."""

        _ = params
        tool_started.set()
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

    async def _run() -> tuple[RunHandle, RunOutcome]:
        handle = await session.prompt("start")
        await _wait_for_provider(provider)
        await tool_started.wait()
        handle.abort()
        report = await handle.wait()
        assert report.status == "aborted"
        return handle, report.outcome

    handle, outcome = asyncio.run(_run())
    history = session.history

    assert outcome == Aborted(reason="cancelled")
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
        assert (await first.wait()).status == "aborted"
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

        first_report = await first.wait()
        assert first_report.status == "aborted"
        assert first_report.outcome == Aborted(reason="replaced")
        second_release.set()
        assert (await second.wait()).status == "completed"
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
        assert (await second.wait()).status == "completed"
        first_release.set()
        first_report = await first.wait()
        assert first_report.status == "aborted"
        assert first_report.outcome == Aborted(reason="replaced")
        assert first_report.persisted
        assert [
            item.content
            for item in second_session.history
            if isinstance(item, UserMessage)
        ] == ["second"]
        first_store.close()
        second_store.close()

    asyncio.run(_run())


def test_start_persistence_failure_raises_before_a_handle_exists() -> None:
    """Propagate Store start failures because no live run was accepted."""

    async def _run() -> None:
        store = SQLiteStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=ProviderStreamMock([]).fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        session = runtime.session(session_id="start-failure")
        store.close()

        with pytest.raises(StorePersistenceError) as raised:
            await session.prompt("cannot start")

        assert raised.value.operation == "start_run"

    asyncio.run(_run())


def test_atomic_finalization_failure_is_visible_and_recoverable() -> None:
    """Report finalization failure without replacing the execution outcome."""

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

        report = await first.wait()

        events = [event async for event in first.events()]
        terminal = events[-1]
        assert isinstance(terminal, RunEndEvent)
        assert report.outcome == Completed(value="lost")
        assert report.status == "completed"
        assert not report.persisted
        assert isinstance(report.finalization_error, StorePersistenceError)
        assert isinstance(report.finalization_error.cause, OSError)
        assert await first.wait() is report
        assert terminal.outcome == Completed(value="lost")
        assert store.get_run(first.id).status == "running"
        assert session.history == ()

        store.fail_finishes = False
        recovery = await session.prompt("second", replace_active=True)
        assert (await recovery.wait()).status == "completed"
        store.close()

    asyncio.run(_run())


def test_agent_failure_survives_a_finalization_failure() -> None:
    """Keep the agent verdict separate from Store finalization diagnostics."""

    store = _FailingFinishStore(in_memory=True)
    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="fail",
                arguments={"reason": "cannot deliver"},
            )
        ]
    )
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        store=store,
    )

    async def _run() -> None:
        handle = await runtime.session(session_id="agent-failure").prompt(
            "fail",
            result=_TextResult,
        )
        report = await handle.wait()

        assert report.outcome == Failed(cause=AgentFailure(reason="cannot deliver"))
        assert report.execution_error is None
        assert isinstance(report.finalization_error, StorePersistenceError)
        assert not report.persisted

    asyncio.run(_run())
    store.close()


def test_execution_failure_survives_a_finalization_failure() -> None:
    """Report both execution and Store errors without conflating their meanings."""

    store = _FailingFinishStore(in_memory=True)
    provider = ProviderStreamMock([error_stream("response-1", "provider unavailable")])
    runtime = AgentRuntime(
        stream_fn=provider.fn,
        model="gpt-5.4",
        cwd=Path("."),
        store=store,
    )

    async def _run() -> None:
        handle = await runtime.session(session_id="execution-failure").prompt("fail")
        report = await handle.wait()

        assert isinstance(report.outcome, Failed)
        assert isinstance(report.outcome.cause, ExecutionFailure)
        assert report.outcome.cause.message == "provider unavailable"
        assert report.execution_error is not None
        assert isinstance(report.finalization_error, StorePersistenceError)

    asyncio.run(_run())
    store.close()


def test_abort_survives_a_finalization_failure() -> None:
    """Preserve explicit cancellation when its terminal write cannot commit."""

    async def _run() -> None:
        release = asyncio.Event()
        provider = GatedProviderStreamMock([release])
        store = _FailingFinishStore(in_memory=True)
        runtime = AgentRuntime(
            stream_fn=provider.fn,
            model="gpt-5.4",
            cwd=Path("."),
            store=store,
        )
        handle = await runtime.session(session_id="abort-failure").prompt("wait")
        await _wait_for_provider(provider)
        handle.abort()
        report = await handle.wait()

        assert report.outcome == Aborted(reason="cancelled")
        assert report.execution_error is None
        assert isinstance(report.finalization_error, StorePersistenceError)
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
        assert (await first.wait()).status == "completed"
        fork = source.fork(session_id="fork", name="Fork")
        assert fork.history == source.history
        assert runtime.runs_for("fork") == ()

        second = await fork.prompt("second")
        assert (await second.wait()).status == "completed"
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

        assert (await handle.wait()).status == "completed"
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
        assert (await first.wait()).status == "completed"
        second = await session.prompt("second")
        assert (await second.wait()).status == "completed"

    asyncio.run(_run())
    replayed = provider.history(1)
    assert len(replayed) == 3
    assert isinstance(replayed[0], UserMessage)
    assert replayed[0].content == "first"
    assert replayed[1].role == "assistant"
    assert isinstance(replayed[2], UserMessage)
    assert replayed[2].content == "second"
    store.close()


def test_run_report_exposes_history_and_latest_assistant_turn() -> None:
    """Expose terminal run-local history through the immutable report."""

    runtime, store, _ = _runtime([final_text_stream("response-1", "done")])

    async def _run() -> None:
        handle = await runtime.session(session_id="output").prompt("hello")
        report = await handle.wait()
        assert report.status == "completed"
        assert report.last_assistant_turn is not None
        assert [item.role for item in report.history_delta] == [
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
        assert (await run.wait()).status == "completed"

    asyncio.run(_run())
    assert captured == [tmp_path.resolve()]
    store.close()


class _FailingFinishStore(SQLiteStore):
    """SQLite Store with a deterministic finalization failure."""

    fail_finishes: bool = True

    def finish_run(
        self,
        *,
        record: RunRecord,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord:
        """Fail or delegate one atomic finish operation."""

        if self.fail_finishes:
            raise StorePersistenceError("finish_run", OSError("disk full"))
        return super().finish_run(
            record=record,
            history_delta=history_delta,
        )


class _UnavailablePublicHistoryStore(SQLiteStore):
    """SQLite Store whose standalone history read is unavailable."""

    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Prove prompt bootstrap does not perform a second Store read."""

        _ = session_id
        raise AssertionError("AgentRuntime must use StartedRun.committed_history")


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

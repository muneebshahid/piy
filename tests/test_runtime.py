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
    AgentHarness,
    Completed,
    ExecutionFailure,
    Failed,
    Faulted,
    HistoryItem,
    Provider,
    RunHandle,
    RunOutcome,
    RunRecord,
    SessionRecord,
    SessionNotFoundError,
    SessionRepository,
    SQLiteStore,
    StorePersistenceError,
)
from tile.events import AgentEvent, RunFaultEvent
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


def test_repository_creates_lists_and_gets_persistent_sessions() -> None:
    """Use the repository as the authoritative session registry."""

    repository = SessionRepository(SQLiteStore(in_memory=True))
    generated = repository.create(name="Generated")
    explicit = repository.create(session_id="known", name="Known")

    assert generated.id != explicit.id
    assert explicit.get_session_record().name == "Known"
    assert [session.id for session in repository.list()] == [generated.id, "known"]
    assert repository.get("known").get_session_record().name == "Known"
    with pytest.raises(SessionNotFoundError, match="missing"):
        repository.get("missing")


def test_runtime_commits_a_complete_turn_only_at_finalization() -> None:
    """Keep prompt and assistant output provisional while the run is active."""

    async def _run() -> None:
        release = asyncio.Event()
        provider = GatedProviderStreamMock([release])
        store = SQLiteStore(in_memory=True)
        session = SessionRepository(store).create(session_id="atomic-history")
        harness = AgentHarness(
            session=session,
            cwd=Path("."),
        )
        configured_provider = provider
        handle = await harness.prompt("hello", provider=configured_provider)
        await _wait_for_provider(provider)

        assert session.get_history() == ()
        assert store.get_run(handle.id).status == "running"
        assert store.get_run(handle.id).prompt == "hello"

        release.set()
        assert await handle.wait() == Completed(value="answer 0")
        prompt = session.get_history()[0]
        assert isinstance(prompt, UserMessage)
        assert prompt.content == "hello"
        assert len(session.get_history()) == 2
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
        session = SessionRepository(store).create(session_id="provisional")
        harness = AgentHarness(
            session=session,
            cwd=Path("."),
            tools=[_tool("inspect", inspect)],
        )
        configured_provider = provider
        handle = await harness.prompt(
            "inspect", provider=configured_provider, result=_TextResult
        )

        await _wait_for_provider(provider, expected=3)
        assert session.get_history() == ()
        assert [item.role for item in provider.history(2)] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
            "user",
        ]

        releases[2].set()
        assert isinstance(await handle.wait(), Completed)
        assert [item.role for item in session.get_history()] == [
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

        class _ObservingProvider(Provider):
            """Read the store at the provider boundary."""

            def __init__(self) -> None:
                """Configure the observing provider."""

                super().__init__(model="gpt-5.4")

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
                """Capture the running record before returning a response."""

                _ = history, instructions, tools
                observed_statuses.extend(
                    run.status for run in store.list_runs("session-1")
                )
                return async_stream(final_text_stream("response-1", "done"))

        session = SessionRepository(store).create(session_id="session-1")
        harness = AgentHarness(session=session, cwd=Path("."))
        provider = _ObservingProvider()
        run = await harness.prompt("hello", provider=provider)
        assert isinstance(await run.wait(), Completed)
        assert observed_statuses == ["running"]
        store.close()

    asyncio.run(_run())


def test_runtime_bootstraps_from_start_run_history_snapshot() -> None:
    """Avoid a second fallible Store read after durable run acceptance."""

    store = _UnavailablePublicHistoryStore(in_memory=True)
    store.create_session(record=SessionRecord.create(id="session-1"))
    started = store.start_run(
        record=RunRecord.start(
            id="seed",
            session_id="session-1",
            prompt="first",
            model="gpt-5.4",
            provider="test",
        ),
    )
    store.finish_run(
        run_id=started.run.id,
        outcome=Completed(value="first answer"),
        history_delta=[
            UserMessage(content="first"),
            AssistantTurn(response_id="response-1"),
        ],
    )
    provider = ProviderStreamMock([final_text_stream("response-2", "second answer")])
    session = SessionRepository(store).get("session-1")
    harness = AgentHarness(session=session, cwd=Path("."))
    configured_provider = provider

    async def _run() -> None:
        handle = await harness.prompt("second", provider=configured_provider)
        assert isinstance(await handle.wait(), Completed)

    asyncio.run(_run())
    assert [item.role for item in provider.history(0)] == [
        "user",
        "assistant",
        "user",
    ]
    store.close()


def test_runtime_failure_commits_the_valid_completed_prefix() -> None:
    """Persist replayable items while excluding an errored assistant turn."""

    harness, store, provider = _harness(
        [error_stream("response-1", "provider unavailable")],
        session_id="failed",
    )
    session = harness.session

    async def _run() -> tuple[RunHandle, RunOutcome]:
        handle = await harness.prompt("hello", provider=_configured(provider))
        outcome = await handle.wait()
        assert isinstance(outcome, Failed)
        return handle, outcome

    handle, outcome = asyncio.run(_run())

    assert outcome == Failed(
        cause=ExecutionFailure(
            origin="turn",
            exception_type="TurnFailedError",
            message="provider unavailable",
        )
    )
    history = session.get_history()
    assert history == (history[0],)
    prompt = history[0]
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
    session = SessionRepository(store).create(session_id="agent-failure-history")
    harness = AgentHarness(session=session, cwd=Path("."))
    configured_provider = provider

    async def _run() -> None:
        handle = await harness.prompt(
            "try", provider=configured_provider, result=_TextResult
        )
        outcome = await handle.wait()

        expected = Failed(cause=AgentFailure(reason="cannot deliver"))
        assert outcome == expected
        assert store.get_run(handle.id).outcome == expected
        assert [item.role for item in session.get_history()] == [
            "user",
            "assistant",
            "tool_result",
        ]

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
    harness, store, provider = _harness(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="blocked",
                arguments={},
            )
        ],
        session_id="abort",
        tools=[tool],
    )
    session = harness.session

    async def _run() -> tuple[RunHandle, RunOutcome]:
        handle = await harness.prompt("start", provider=_configured(provider))
        await _wait_for_provider(provider)
        await tool_started.wait()
        handle.abort()
        outcome = await handle.wait()
        assert isinstance(outcome, Aborted)
        return handle, outcome

    handle, outcome = asyncio.run(_run())
    history = session.get_history()

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
        session = SessionRepository(store).create(session_id="busy")
        harness = AgentHarness(session=session, cwd=Path("."))
        configured_provider = provider
        first = await harness.prompt("first", provider=configured_provider)
        await _wait_for_provider(provider)

        with pytest.raises(ActiveRunError, match="busy"):
            await harness.prompt("second", provider=configured_provider)

        first.abort()
        assert await first.wait() == Aborted(reason="cancelled")
        store.close()

    asyncio.run(_run())


def test_durable_abort_fences_old_history_and_runs_the_successor() -> None:
    """Fence a local predecessor in storage and commit only the new turn."""

    async def _run() -> None:
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        provider = GatedProviderStreamMock([first_release, second_release])
        store = SQLiteStore(in_memory=True)
        repository = SessionRepository(store)
        session = repository.create(session_id="recover")
        harness = AgentHarness(session=session, cwd=Path("."))
        configured_provider = provider
        first = await harness.prompt("first", provider=configured_provider)
        await _wait_for_provider(provider)
        aborted = repository.abort_active_run(session.id)
        second = await harness.prompt("second", provider=configured_provider)
        await _wait_for_provider(provider, expected=2)

        second_release.set()
        assert isinstance(await second.wait(), Completed)
        first_release.set()
        assert await first.wait() == Aborted(reason="recovered")
        assert aborted is not None
        assert aborted.outcome == Aborted(reason="recovered")
        assert [
            item.content
            for item in session.get_history()
            if isinstance(item, UserMessage)
        ] == ["second"]
        assert store.get_run(first.id).outcome == Aborted(reason="recovered")
        store.close()

    asyncio.run(_run())


def test_durable_abort_works_across_harness_instances(tmp_path: Path) -> None:
    """Use repository fencing when the predecessor belongs to another harness."""

    async def _run() -> None:
        database_path = tmp_path / "shared.db"
        first_store = SQLiteStore(database_path)
        second_store = SQLiteStore(database_path)
        first_release = asyncio.Event()
        first_provider = GatedProviderStreamMock([first_release])
        second_provider = ProviderStreamMock(
            [final_text_stream("response-2", "replacement")]
        )
        first_repository = SessionRepository(first_store)
        first_session = first_repository.create(session_id="shared")
        first_harness = AgentHarness(session=first_session, cwd=Path("."))
        first = await first_harness.prompt(
            "first", provider=_configured(first_provider)
        )
        await _wait_for_provider(first_provider)

        second_repository = SessionRepository(second_store)
        second_session = second_repository.get("shared")
        second_harness = AgentHarness(session=second_session, cwd=Path("."))
        aborted = second_repository.abort_active_run(second_session.id)
        second = await second_harness.prompt(
            "second",
            provider=_configured(second_provider),
        )
        assert isinstance(await second.wait(), Completed)
        first_release.set()
        assert await first.wait() == Aborted(reason="recovered")
        assert aborted is not None
        assert aborted.id == first.id
        assert [
            item.content
            for item in second_session.get_history()
            if isinstance(item, UserMessage)
        ] == ["second"]
        first_store.close()
        second_store.close()

    asyncio.run(_run())


def test_start_persistence_failure_raises_before_a_handle_exists() -> None:
    """Propagate Store start failures because no live run was accepted."""

    async def _run() -> None:
        store = SQLiteStore(in_memory=True)
        transport = ProviderStreamMock([])
        session = SessionRepository(store).create(session_id="start-failure")
        harness = AgentHarness(session=session, cwd=Path("."))
        store.close()

        with pytest.raises(StorePersistenceError) as raised:
            await harness.prompt("cannot start", provider=_configured(transport))

        assert raised.value.operation == "start_run"

    asyncio.run(_run())


def test_atomic_finalization_failure_faults_handle_until_durable_recovery() -> None:
    """Expose a finalization fault and reuse the harness after recovery."""

    async def _run() -> None:
        store = _FailingFinishStore(in_memory=True)
        provider = ProviderStreamMock(
            [
                final_text_stream("response-1", "lost"),
                final_text_stream("response-2", "recovered"),
            ]
        )
        repository = SessionRepository(store)
        session = repository.create(session_id="failure")
        harness = AgentHarness(session=session, cwd=Path("."))
        configured_provider = _configured(provider)
        first = await harness.prompt("first", provider=configured_provider)

        result = await first.wait()

        events = [event async for event in first.events()]
        terminal = events[-1]
        assert isinstance(terminal, RunFaultEvent)
        assert isinstance(result, Faulted)
        assert isinstance(result.error, StorePersistenceError)
        assert isinstance(result.error.cause, OSError)
        assert await first.wait() is result
        assert store.get_run(first.id).status == "running"
        assert session.get_history() == ()

        store.fail_finishes = False
        aborted = repository.abort_active_run(session.id)
        assert aborted is not None
        assert aborted.outcome == Aborted(reason="recovered")
        recovery = await harness.prompt("second", provider=configured_provider)
        assert isinstance(await recovery.wait(), Completed)
        store.close()

    asyncio.run(_run())


def test_agent_failure_is_overridden_by_a_finalization_fault() -> None:
    """Return the durability fault when an agent failure cannot be persisted."""

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
    session = SessionRepository(store).create(session_id="agent-failure")
    harness = AgentHarness(session=session, cwd=Path("."))

    async def _run() -> None:
        handle = await harness.prompt(
            "fail",
            provider=_configured(provider),
            result=_TextResult,
        )
        result = await handle.wait()

        assert isinstance(result, Faulted)
        assert isinstance(result.error, StorePersistenceError)

    asyncio.run(_run())
    store.close()


def test_execution_failure_is_overridden_by_a_finalization_fault() -> None:
    """Return the durability fault when execution failure cannot be persisted."""

    store = _FailingFinishStore(in_memory=True)
    provider = ProviderStreamMock([error_stream("response-1", "provider unavailable")])
    session = SessionRepository(store).create(session_id="execution-failure")
    harness = AgentHarness(session=session, cwd=Path("."))

    async def _run() -> None:
        handle = await harness.prompt("fail", provider=_configured(provider))
        result = await handle.wait()

        assert isinstance(result, Faulted)
        assert isinstance(result.error, StorePersistenceError)

    asyncio.run(_run())
    store.close()


def test_abort_is_overridden_by_a_finalization_fault() -> None:
    """Return the durability fault when cancellation cannot be persisted."""

    async def _run() -> None:
        release = asyncio.Event()
        provider = GatedProviderStreamMock([release])
        store = _FailingFinishStore(in_memory=True)
        session = SessionRepository(store).create(session_id="abort-failure")
        harness = AgentHarness(session=session, cwd=Path("."))
        handle = await harness.prompt("wait", provider=_configured(provider))
        await _wait_for_provider(provider)
        handle.abort()
        result = await handle.wait()

        assert isinstance(result, Faulted)
        assert isinstance(result.error, StorePersistenceError)
        store.close()

    asyncio.run(_run())


def test_forked_session_inherits_flat_history_and_diverges() -> None:
    """Fork committed history without copying run ownership."""

    provider = ProviderStreamMock(
        [
            final_text_stream("response-1", "source"),
            final_text_stream("response-2", "fork"),
        ]
    )
    store = SQLiteStore(in_memory=True)
    repository = SessionRepository(store)
    source = repository.create(session_id="source")
    source_harness = AgentHarness(session=source, cwd=Path("."))
    configured_provider = _configured(provider)

    async def _run() -> None:
        first = await source_harness.prompt("first", provider=configured_provider)
        assert isinstance(await first.wait(), Completed)
        fork = repository.fork("source", target_session_id="fork", name="Fork")
        assert fork.get_history() == source.get_history()
        assert fork.get_runs() == ()

        fork_harness = AgentHarness(session=fork, cwd=Path("."))
        second = await fork_harness.prompt("second", provider=configured_provider)
        assert isinstance(await second.wait(), Completed)
        assert len(fork.get_history()) == 4
        assert len(source.get_history()) == 2

    asyncio.run(_run())
    store.close()


def test_multiple_subscribers_replay_the_same_closed_log() -> None:
    """Keep event subscription independent from execution ownership."""

    harness, store, provider = _harness(
        [final_text_stream("response-1", "done")],
        session_id="subscribers",
    )

    async def _run() -> None:
        handle = await harness.prompt("hello", provider=_configured(provider))
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

    harness, store, provider = _harness(
        [final_text_stream("response-1", "done")],
        session_id="early-stop",
    )
    session = harness.session

    async def _run() -> None:
        handle = await harness.prompt("hello", provider=_configured(provider))
        async for _ in handle.events():
            break

        assert isinstance(await handle.wait(), Completed)
        assert len(session.get_history()) == 2

    asyncio.run(_run())
    store.close()


def test_next_prompt_replays_committed_history_and_current_prompt() -> None:
    """Build later requests from typed committed history plus the new prompt."""

    harness, store, provider = _harness(
        [
            final_text_stream("response-1", "first answer"),
            final_text_stream("response-2", "second answer"),
        ],
        session_id="multi-turn",
    )
    configured_provider = _configured(provider)

    async def _run() -> None:
        first = await harness.prompt("first", provider=configured_provider)
        assert isinstance(await first.wait(), Completed)
        second = await harness.prompt("second", provider=configured_provider)
        assert isinstance(await second.wait(), Completed)

    asyncio.run(_run())
    replayed = provider.history(1)
    assert len(replayed) == 3
    assert isinstance(replayed[0], UserMessage)
    assert replayed[0].content == "first"
    assert replayed[1].role == "assistant"
    assert isinstance(replayed[2], UserMessage)
    assert replayed[2].content == "second"
    store.close()


def test_wait_returns_only_outcome_while_session_exposes_history() -> None:
    """Keep run results small and expose durable history through the session."""

    harness, store, provider = _harness(
        [final_text_stream("response-1", "done")],
        session_id="output",
    )

    async def _run() -> None:
        handle = await harness.prompt("hello", provider=_configured(provider))
        assert await handle.wait() == Completed(value="done")
        assert [item.role for item in harness.session.get_history()] == [
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
    harness, store, provider = _harness(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="inspect_cwd",
                arguments={},
            ),
            final_text_stream("response-2", "done"),
        ],
        session_id="cwd",
        tools=[valid],
        cwd=tmp_path,
    )

    async def _run() -> None:
        run = await harness.prompt("inspect", provider=_configured(provider))
        assert isinstance(await run.wait(), Completed)

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
    ) -> RunRecord:
        """Fail or delegate one atomic finish operation."""

        if self.fail_finishes:
            raise StorePersistenceError("finish_run", OSError("disk full"))
        return super().finish_run(
            run_id=run_id,
            outcome=outcome,
            history_delta=history_delta,
        )


class _UnavailablePublicHistoryStore(SQLiteStore):
    """SQLite Store whose standalone history read is unavailable."""

    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Prove prompt bootstrap does not perform a second Store read."""

        _ = session_id
        raise AssertionError("RunExecution must use StartedRun.committed_history")


def _harness(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    *,
    session_id: str,
    tools: Sequence[ToolDefinition] = (),
    cwd: Path = Path("."),
) -> tuple[AgentHarness, SQLiteStore, ProviderStreamMock]:
    """Build a session-bound harness with an in-memory SQLite Store."""

    provider = ProviderStreamMock(streams)
    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create(session_id=session_id)
    harness = AgentHarness(
        session=session,
        cwd=cwd,
        tools=tools,
    )
    return harness, store, provider


def _configured(provider: ProviderStreamMock) -> Provider:
    """Bind a deterministic transport to the test model."""

    return provider


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

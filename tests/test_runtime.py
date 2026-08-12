"""End-to-end tests for the persistence-first runtime and its harness API."""

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import override

import pytest
from pydantic import BaseModel

from tests.support.agent_streams import (
    GatedProviderStreamMock,
    GatedQueuedProviderStreamMock,
    ProviderStreamMock,
    error_stream,
    final_text_stream,
    tool_call_stream,
)
from tests.support.async_streams import async_stream
from tests.support.harnesses import build_harness
from tests.support.store import (
    FailingFinishStore,
    FailingStartStore,
    create_session,
    persist_outcome,
    start_run,
    terminal_outcome,
)
from tests.support.tool_definitions import NoInput
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


class _TextResult(BaseModel):
    """Minimal typed result for explicit agent-failure tests."""

    value: str


@pytest.fixture
def failing_start_store() -> Iterator[FailingStartStore]:
    """Yield an in-memory store with a recoverable admission failure."""

    store = FailingStartStore(in_memory=True)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def failing_finish_store() -> Iterator[FailingFinishStore]:
    """Yield an in-memory store with a recoverable finalization failure."""

    store = FailingFinishStore(in_memory=True)
    try:
        yield store
    finally:
        store.close()


async def test_harness_runs_prompts_for_its_single_session(
    store: SQLiteStore,
) -> None:
    """Execute through the target repository, session, provider, and harness API."""

    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "done")])
    harness = AgentHarness(
        session=session,
        cwd=Path(),
        instructions="Test agent.",
    )

    handle = await harness.prompt("hello", provider=transport)
    result = await handle.wait()

    assert result == Completed(value="done")
    assert harness.session is session
    assert terminal_outcome(session.get_runs()[0]) == result
    assert [item.role for item in session.get_history()] == ["user", "assistant"]


async def test_harness_accepts_a_different_configured_provider_per_prompt(
    store: SQLiteStore,
) -> None:
    """Persist each prompt's effective provider model instead of harness config."""

    first_transport = ProviderStreamMock(
        [final_text_stream("response-1", "first")],
        model="model-a",
    )
    second_transport = ProviderStreamMock(
        [final_text_stream("response-2", "second")],
        model="model-b",
    )
    harness = build_harness(store, session_id="session-1")

    first = await harness.prompt("first", provider=first_transport)
    assert await first.wait() == Completed(value="first")
    second = await harness.prompt("second", provider=second_transport)
    assert await second.wait() == Completed(value="second")

    assert [record.model for record in harness.session.get_runs()] == [
        "model-a",
        "model-b",
    ]


async def test_runtime_keeps_multi_attempt_history_provisional_until_finalization(
    store: SQLiteStore,
) -> None:
    """Keep history local while the durable record stays active with its prompt."""

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

    async def inspect(params: NoInput) -> ToolResult:
        """Return one replayable non-terminal tool result."""

        _ = params
        return ToolResult.text("inspected")

    harness = build_harness(
        store,
        session_id="provisional",
        tools=[_tool("inspect", inspect)],
    )
    session = harness.session
    handle = await harness.prompt("inspect", provider=provider, result_type=_TextResult)

    await provider.wait_for_calls(expected=3)
    assert session.get_history() == ()
    assert store.get_run(session_id=session.id, run_id=handle.id).status == "active"
    assert store.get_run(session_id=session.id, run_id=handle.id).prompt == "inspect"
    assert [item.role for item in provider.history(2)] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
        "user",
    ]

    releases[2].set()
    assert await handle.wait() == Completed(value=_TextResult(value="done"))
    assert terminal_outcome(
        store.get_run(session_id=session.id, run_id=handle.id)
    ) == Completed(value={"value": "done"})
    assert [item.role for item in session.get_history()] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
        "user",
        "assistant",
        "tool_result",
    ]


async def test_runtime_persists_active_record_before_provider_execution(
    store: SQLiteStore,
) -> None:
    """Make the submitted prompt durable before invoking the provider."""

    observed_statuses: list[str] = []

    class _ObservingProvider(Provider):
        """Read the store at the provider boundary."""

        def __init__(self) -> None:
            """Configure the observing provider."""

            super().__init__(model="gpt-5.4")

        @property
        @override
        def name(self) -> str:
            """Return the deterministic provider identity."""

            return "test"

        @override
        async def stream(
            self,
            history: Sequence[ConversationItem],
            *,
            instructions: str,
            tools: Sequence[ToolDefinition] | None,
        ) -> AsyncEventStream:
            """Capture the active record before returning a response."""

            _ = history, instructions, tools
            observed_statuses.extend(run.status for run in store.list_runs("session-1"))
            return async_stream(final_text_stream("response-1", "done"))

    harness = build_harness(store, session_id="session-1")
    run = await harness.prompt("hello", provider=_ObservingProvider())

    assert isinstance(await run.wait(), Completed)
    assert observed_statuses == ["active"]


async def test_runtime_bootstraps_from_start_run_history_snapshot() -> None:
    """Avoid a second fallible Store read after durable run acceptance."""

    store = _UnavailablePublicHistoryStore(in_memory=True)
    try:
        create_session(store, session_id="session-1")
        started = start_run(store, run_id="seed", prompt="first")
        persist_outcome(
            store,
            outcome=Completed(value="first answer"),
            history_delta=[
                UserMessage(content="first"),
                AssistantTurn(response_id="response-1"),
            ],
            run_id=started.run.id,
        )
        provider = ProviderStreamMock(
            [final_text_stream("response-2", "second answer")]
        )
        session = SessionRepository(store).get("session-1")
        harness = AgentHarness(
            session=session,
            cwd=Path(),
            instructions="Test agent.",
        )

        handle = await harness.prompt("second", provider=provider)

        assert isinstance(await handle.wait(), Completed)
        assert [item.role for item in provider.history(0)] == [
            "user",
            "assistant",
            "user",
        ]
    finally:
        store.close()


async def test_runtime_failure_commits_the_valid_completed_prefix(
    store: SQLiteStore,
) -> None:
    """Persist replayable items while excluding an errored assistant turn."""

    provider = ProviderStreamMock([error_stream("response-1", "provider unavailable")])
    harness = build_harness(store, session_id="failed")
    session = harness.session

    handle = await harness.prompt("hello", provider=provider)
    outcome = await handle.wait()

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


async def test_agent_failure_commits_its_complete_replayable_history(
    store: SQLiteStore,
) -> None:
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
    harness = build_harness(store, session_id="agent-failure-history")
    session = harness.session

    handle = await harness.prompt("try", provider=provider, result_type=_TextResult)
    outcome = await handle.wait()

    expected = Failed(cause=AgentFailure(reason="cannot deliver"))
    assert outcome == expected
    assert (
        terminal_outcome(store.get_run(session_id=session.id, run_id=handle.id))
        == expected
    )
    assert [item.role for item in session.get_history()] == [
        "user",
        "assistant",
        "tool_result",
    ]


async def test_abort_commits_a_healed_replayable_prefix(store: SQLiteStore) -> None:
    """Heal an interrupted tool call before atomically committing an abort."""

    tool_started = asyncio.Event()

    async def _blocked_tool(params: NoInput) -> ToolResult:
        """Block until task cancellation interrupts execution."""

        _ = params
        tool_started.set()
        await asyncio.Event().wait()
        return ToolResult.text("unreachable")

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="blocked",
                arguments={},
            )
        ]
    )
    harness = build_harness(
        store,
        session_id="abort",
        tools=[_tool("blocked", _blocked_tool)],
    )
    session = harness.session

    handle = await harness.prompt("start", provider=provider)
    await provider.wait_for_calls()
    await tool_started.wait()
    handle.abort()
    outcome = await handle.wait()

    assert outcome == Aborted(reason="cancelled")
    history = session.get_history()
    assert len(history) == 3
    healed = history[-1]
    assert isinstance(healed, ToolResultTurn)
    assert healed.is_error


async def test_overlapping_prompt_is_rejected_by_the_store(
    store: SQLiteStore,
) -> None:
    """Enforce one active run without a process-local session lock."""

    release = asyncio.Event()
    provider = GatedProviderStreamMock([release])
    harness = build_harness(store, session_id="busy")
    first = await harness.prompt("first", provider=provider)
    await provider.wait_for_calls()

    with pytest.raises(ActiveRunError, match="busy"):
        await harness.prompt("second", provider=provider)

    first.abort()
    assert await first.wait() == Aborted(reason="cancelled")


async def test_durable_abort_fences_old_history_and_runs_the_successor(
    store: SQLiteStore,
) -> None:
    """Fence a local predecessor in storage and commit only the new turn."""

    first_release = asyncio.Event()
    second_release = asyncio.Event()
    provider = GatedProviderStreamMock([first_release, second_release])
    harness = build_harness(store, session_id="escape-hatch")
    session = harness.session
    repository = SessionRepository(store)
    first = await harness.prompt("first", provider=provider)
    await provider.wait_for_calls()
    aborted = repository.abort_active_run(session.id)
    second = await harness.prompt("second", provider=provider)
    await provider.wait_for_calls(expected=2)

    second_release.set()
    assert isinstance(await second.wait(), Completed)
    first_release.set()
    assert await first.wait() == Aborted(reason="cancelled")
    assert aborted is not None
    assert aborted.outcome == Aborted(reason="cancelled")
    assert [
        item.content for item in session.get_history() if isinstance(item, UserMessage)
    ] == ["second"]
    assert terminal_outcome(
        store.get_run(session_id=session.id, run_id=first.id)
    ) == Aborted(reason="cancelled")


async def test_durable_abort_works_across_harness_instances(tmp_path: Path) -> None:
    """Use repository fencing when the predecessor belongs to another harness."""

    database_path = tmp_path / "shared.db"
    first_store = SQLiteStore(database_path)
    second_store = SQLiteStore(database_path)
    try:
        first_release = asyncio.Event()
        first_provider = GatedProviderStreamMock([first_release])
        second_provider = ProviderStreamMock([final_text_stream("response-2", "retry")])
        first_harness = build_harness(first_store, session_id="shared")
        first = await first_harness.prompt("first", provider=first_provider)
        await first_provider.wait_for_calls()

        second_repository = SessionRepository(second_store)
        second_session = second_repository.get("shared")
        second_harness = AgentHarness(
            session=second_session,
            cwd=Path(),
            instructions="Test agent.",
        )
        aborted = second_repository.abort_active_run(second_session.id)
        second = await second_harness.prompt("second", provider=second_provider)

        assert isinstance(await second.wait(), Completed)
        first_release.set()
        assert await first.wait() == Aborted(reason="cancelled")
        assert aborted is not None
        assert aborted.id == first.id
        assert [
            item.content
            for item in second_session.get_history()
            if isinstance(item, UserMessage)
        ] == ["second"]
    finally:
        first_store.close()
        second_store.close()


async def test_start_persistence_failure_does_not_disable_the_harness(
    failing_start_store: FailingStartStore,
) -> None:
    """Propagate the admission failure, then reuse the harness once resolved."""

    provider = ProviderStreamMock([final_text_stream("response-1", "recovered")])
    harness = build_harness(failing_start_store, session_id="session-1")

    with pytest.raises(StorePersistenceError) as raised:
        await harness.prompt("first", provider=provider)

    assert raised.value.operation == "start_run"
    failing_start_store.fail_starts = False
    handle = await harness.prompt("second", provider=provider)

    assert await handle.wait() == Completed(value="recovered")
    assert harness.session.get_runs()[0].prompt == "second"


async def test_finalization_fault_requires_the_durable_abort_escape_hatch(
    failing_finish_store: FailingFinishStore,
) -> None:
    """Unblock the harness explicitly after a terminal persistence failure."""

    provider = ProviderStreamMock(
        [
            final_text_stream("response-1", "lost"),
            final_text_stream("response-2", "retry"),
        ]
    )
    harness = build_harness(failing_finish_store, session_id="failure")
    session = harness.session
    repository = SessionRepository(failing_finish_store)
    first = await harness.prompt("first", provider=provider)

    result = await first.wait()

    events = [event async for event in first.events()]
    assert isinstance(events[-1], RunFaultEvent)
    assert isinstance(result, Faulted)
    assert isinstance(result.error, StorePersistenceError)
    assert isinstance(result.error.cause, OSError)
    assert await first.wait() is result
    assert (
        failing_finish_store.get_run(session_id=session.id, run_id=first.id).status
        == "active"
    )
    assert session.get_history() == ()
    with pytest.raises(ActiveRunError):
        await harness.prompt("blocked", provider=provider)

    failing_finish_store.fail_finishes = False
    aborted = repository.abort_active_run(session.id)
    assert aborted is not None
    assert aborted.outcome == Aborted(reason="cancelled")
    retry = await harness.prompt("second", provider=provider)
    assert isinstance(await retry.wait(), Completed)


@pytest.mark.parametrize(
    ("streams", "result_type"),
    [
        pytest.param(
            [
                tool_call_stream(
                    response_id="response-1",
                    call_id="call-1",
                    tool_name="fail",
                    arguments={"reason": "cannot deliver"},
                )
            ],
            _TextResult,
            id="agent-failure",
        ),
        pytest.param(
            [error_stream("response-1", "provider unavailable")],
            None,
            id="execution-failure",
        ),
    ],
)
async def test_failure_is_overridden_by_a_finalization_fault(
    failing_finish_store: FailingFinishStore,
    streams: Sequence[Sequence[ProviderStreamEvent]],
    result_type: type[BaseModel] | None,
) -> None:
    """Return the durability fault when a failure outcome cannot be persisted."""

    provider = ProviderStreamMock(streams)
    harness = build_harness(failing_finish_store, session_id="failure-override")

    handle = await harness.prompt("fail", provider=provider, result_type=result_type)
    outcome = await handle.wait()

    assert isinstance(outcome, Faulted)
    assert isinstance(outcome.error, StorePersistenceError)


async def test_abort_is_overridden_by_a_finalization_fault(
    failing_finish_store: FailingFinishStore,
) -> None:
    """Return the durability fault when cancellation cannot be persisted."""

    release = asyncio.Event()
    provider = GatedProviderStreamMock([release])
    harness = build_harness(failing_finish_store, session_id="abort-failure")
    handle = await harness.prompt("wait", provider=provider)
    await provider.wait_for_calls()
    handle.abort()
    result = await handle.wait()

    assert isinstance(result, Faulted)
    assert isinstance(result.error, StorePersistenceError)


async def test_forked_session_inherits_flat_history_and_diverges(
    store: SQLiteStore,
) -> None:
    """Fork committed history without copying run ownership."""

    provider = ProviderStreamMock(
        [
            final_text_stream("response-1", "source"),
            final_text_stream("response-2", "fork"),
        ]
    )
    source_harness = build_harness(store, session_id="source")
    source = source_harness.session
    repository = SessionRepository(store)

    first = await source_harness.prompt("first", provider=provider)
    assert isinstance(await first.wait(), Completed)
    fork = repository.fork("source", target_session_id="fork")
    assert fork.get_history() == source.get_history()
    assert fork.get_runs() == ()

    fork_harness = AgentHarness(
        session=fork,
        cwd=Path(),
        instructions="Test agent.",
    )
    second = await fork_harness.prompt("second", provider=provider)
    assert isinstance(await second.wait(), Completed)
    assert len(fork.get_history()) == 4
    assert len(source.get_history()) == 2


async def test_multiple_subscribers_replay_the_same_closed_log(
    store: SQLiteStore,
) -> None:
    """Keep event subscription independent from execution ownership."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    harness = build_harness(store, session_id="subscribers")

    handle = await harness.prompt("hello", provider=provider)
    first, second = await asyncio.gather(
        _collect_events(handle),
        _collect_events(handle),
    )

    assert first == second
    assert first[-1].type == "run_end"


async def test_run_continues_after_a_subscriber_stops_consuming(
    store: SQLiteStore,
) -> None:
    """Keep execution task-owned when an event subscriber disconnects."""

    provider = ProviderStreamMock([final_text_stream("response-1", "done")])
    harness = build_harness(store, session_id="early-stop")

    handle = await harness.prompt("hello", provider=provider)
    async for _ in handle.events():
        break

    assert isinstance(await handle.wait(), Completed)
    assert len(harness.session.get_history()) == 2


async def test_next_prompt_replays_committed_history_and_current_prompt(
    store: SQLiteStore,
) -> None:
    """Build later requests from typed committed history plus the new prompt."""

    provider = ProviderStreamMock(
        [
            final_text_stream("response-1", "first answer"),
            final_text_stream("response-2", "second answer"),
        ]
    )
    harness = build_harness(store, session_id="multi-turn")

    first = await harness.prompt("first", provider=provider)
    assert isinstance(await first.wait(), Completed)
    second = await harness.prompt("second", provider=provider)
    assert isinstance(await second.wait(), Completed)

    replayed = provider.history(1)
    assert len(replayed) == 3
    assert isinstance(replayed[0], UserMessage)
    assert replayed[0].content == "first"
    assert replayed[1].role == "assistant"
    assert isinstance(replayed[2], UserMessage)
    assert replayed[2].content == "second"


async def test_runtime_binds_cwd_into_tool_functions(
    store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Inject the harness cwd into tool functions declaring a cwd parameter."""

    captured: list[Path] = []

    async def inspect_cwd(params: NoInput, *, cwd: Path) -> ToolResult:
        """Capture the injected path."""

        _ = params
        captured.append(cwd)
        return ToolResult.text("ok")

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="response-1",
                call_id="call-1",
                tool_name="inspect_cwd",
                arguments={},
            ),
            final_text_stream("response-2", "done"),
        ]
    )
    harness = build_harness(
        store,
        session_id="cwd",
        tools=[_tool("inspect_cwd", inspect_cwd)],
        cwd=tmp_path,
    )

    run = await harness.prompt("inspect", provider=provider)

    assert isinstance(await run.wait(), Completed)
    assert captured == [tmp_path.resolve()]


def test_runtime_rejects_model_visible_cwd_schema_property(
    store: SQLiteStore,
) -> None:
    """Reject harness construction for a tool exposing cwd in its input schema."""

    class _CwdInput(ToolInput):
        """Input schema that illegally exposes the harness-injected cwd."""

        cwd: str

    async def shadowing_cwd(params: _CwdInput, *, cwd: Path) -> ToolResult:
        """Fail loudly if the harness accepts a model-visible cwd tool."""

        _ = params, cwd
        raise AssertionError("rejected tool must never execute")

    tool = ToolDefinition(
        name="shadowing_cwd",
        description="Exercise shadowing_cwd.",
        input_model=_CwdInput,
        fn=shadowing_cwd,
    )

    with pytest.raises(ValueError, match="declares cwd in its input schema"):
        build_harness(store, session_id="cwd-schema", tools=[tool])


class _UnavailablePublicHistoryStore(SQLiteStore):
    """SQLite Store whose standalone history read is unavailable."""

    @override
    def get_history(self, session_id: str) -> tuple[HistoryItem, ...]:
        """Prove prompt bootstrap does not perform a second Store read."""

        _ = session_id
        raise AssertionError("RunExecution must use StartedRun.committed_history")


async def _collect_events(handle: RunHandle) -> list[AgentEvent]:
    """Collect one complete live run log."""

    return [event async for event in handle.events()]


def _tool(name: str, fn: ToolFunction) -> ToolDefinition:
    """Build a no-input test tool."""

    return ToolDefinition(
        name=name,
        description=f"Exercise {name}.",
        input_model=NoInput,
        fn=fn,
    )

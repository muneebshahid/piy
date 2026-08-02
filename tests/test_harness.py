"""End-to-end tests for the single-session AgentHarness API."""

import asyncio
from pathlib import Path

import pytest

from tile import (
    Aborted,
    ActiveRunError,
    AgentHarness,
    Completed,
    Faulted,
    Provider,
    RunOutcome,
    RunRecord,
    SessionRepository,
    SQLiteStore,
    StorePersistenceError,
)
from tile.types import UserMessage
from tests.support.agent_streams import (
    GatedProviderStreamMock,
    ProviderStreamMock,
    final_text_stream,
)
from tests.support.store import FailingFinishStore, FailingStartStore


def test_harness_runs_prompts_for_its_single_session() -> None:
    """Execute through the target repository, session, provider, and harness API."""

    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    transport = ProviderStreamMock([final_text_stream("response-1", "done")])
    harness = AgentHarness(session=session, cwd=Path("."))

    async def run() -> RunOutcome | Faulted:
        """Prompt the harness and wait for its terminal result."""

        handle = await harness.prompt("hello", provider=_provider(transport))
        return await handle.wait()

    result = asyncio.run(run())

    assert result == Completed(value="done")
    assert harness.session is session
    assert session.get_runs()[0].outcome == result
    assert [item.role for item in session.get_history()] == ["user", "assistant"]
    store.close()


def test_harness_accepts_a_different_configured_provider_per_prompt() -> None:
    """Persist each prompt's effective provider model instead of harness config."""

    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    first_transport = ProviderStreamMock(
        [final_text_stream("response-1", "first")],
        model="model-a",
    )
    second_transport = ProviderStreamMock(
        [final_text_stream("response-2", "second")],
        model="model-b",
    )
    harness = AgentHarness(session=session, cwd=Path("."))

    async def run() -> None:
        """Complete sequential prompts through different provider values."""

        first = await harness.prompt(
            "first",
            provider=_provider(first_transport),
        )
        assert await first.wait() == Completed(value="first")
        second = await harness.prompt(
            "second",
            provider=_provider(second_transport),
        )
        assert await second.wait() == Completed(value="second")

    asyncio.run(run())

    assert [record.model for record in session.get_runs()] == ["model-a", "model-b"]
    store.close()


def test_repository_escape_hatch_fences_a_local_run_and_unblocks_harness() -> None:
    """Reconcile local work to a durable abort before running a successor."""

    store = SQLiteStore(in_memory=True)
    repository = SessionRepository(store)
    session = repository.create(session_id="session-1")

    async def run() -> tuple[
        RunRecord | None, RunOutcome | Faulted, RunOutcome | Faulted
    ]:
        """Abort a blocked durable record and wait for both terminal results."""

        release = asyncio.Event()
        first_transport = GatedProviderStreamMock([release])
        second_transport = ProviderStreamMock(
            [final_text_stream("response-2", "retry")]
        )
        harness = AgentHarness(session=session, cwd=Path("."))
        first = await harness.prompt("first", provider=_provider(first_transport))
        await _wait_for_provider(first_transport)
        aborted = repository.abort_active_run(session.id)
        second = await harness.prompt(
            "second",
            provider=_provider(second_transport),
        )
        release.set()
        return aborted, await first.wait(), await second.wait()

    aborted, first_result, second_result = asyncio.run(run())

    assert aborted is not None
    assert aborted.outcome == Aborted(reason="cancelled")
    assert first_result == Aborted(reason="cancelled")
    assert second_result == Completed(value="retry")
    assert [
        item.content for item in session.get_history() if isinstance(item, UserMessage)
    ] == ["second"]
    store.close()


def test_start_persistence_failure_does_not_disable_the_harness() -> None:
    """Reuse the same harness after a transient admission failure is resolved."""

    store = FailingStartStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    harness = AgentHarness(session=session, cwd=Path("."))
    provider = _provider(
        ProviderStreamMock([final_text_stream("response-1", "recovered")])
    )

    async def run() -> RunOutcome | Faulted:
        """Retry through the same harness after restoring Store admission."""

        with pytest.raises(StorePersistenceError):
            await harness.prompt("first", provider=provider)
        store.fail_starts = False
        handle = await harness.prompt("second", provider=provider)
        return await handle.wait()

    result = asyncio.run(run())

    assert result == Completed(value="recovered")
    assert session.get_runs()[0].prompt == "second"
    store.close()


def test_finalization_failure_leaves_admission_blocked_by_the_store() -> None:
    """Return Faulted while the durable running record fences another prompt."""

    store = FailingFinishStore(in_memory=True)
    session = SessionRepository(store).create(session_id="session-1")
    harness = AgentHarness(session=session, cwd=Path("."))
    provider = _provider(ProviderStreamMock([final_text_stream("response-1", "lost")]))

    async def run() -> RunOutcome | Faulted:
        """Finish one run and attempt another after its durability fault."""

        handle = await harness.prompt("first", provider=provider)
        result = await handle.wait()
        with pytest.raises(ActiveRunError):
            await harness.prompt("second", provider=provider)
        return result

    result = asyncio.run(run())

    assert isinstance(result, Faulted)
    assert isinstance(result.error, StorePersistenceError)
    assert session.get_runs()[0].status == "running"
    store.close()


def _provider(
    transport: ProviderStreamMock,
) -> Provider:
    """Configure one fake provider transport."""

    return transport


async def _wait_for_provider(provider: ProviderStreamMock) -> None:
    """Wait for one deterministic provider invocation."""

    for _ in range(30):
        if provider.await_count:
            return
        await asyncio.sleep(0)
    raise AssertionError("Expected one provider invocation.")

"""Tests for the example local headless runner."""

import asyncio
import io
import json
from pathlib import Path

import pytest

from tile import Completed, Provider, RunResult
from tile.types.conversation import ConversationItem
from tile.types.tools import ToolTextContent
from examples import local_runner
from examples.local_runner import run_cli, run_prompt
from tests.support.agent_streams import (
    ProviderStreamMock,
    final_text_stream,
    tool_call_stream,
)
from tests.support.conversation_assertions import (
    expect_tool_result_turn,
    expect_user_message,
)


def test_run_prompt_streams_runtime_tool_flow_as_json_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one prompt through the local runtime with a deterministic file tool."""

    provider, output = _run_runtime_tool_flow(tmp_path, monkeypatch)

    _assert_runtime_event_sequence(output)
    _assert_provider_received_tool_result(provider)


def test_run_cli_rejects_empty_prompt() -> None:
    """Reject a missing prompt before constructing the default agent."""

    status = asyncio.run(run_cli(["   "]))

    assert status == 2


def test_run_cli_reads_prompt_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read a prompt from standard input when no prompt arguments are supplied."""

    prompts: list[str] = []

    async def _record_prompt(prompt: str, *, provider: Provider) -> RunResult:
        """Record the prompt passed by the CLI."""

        _ = provider
        prompts.append(prompt)
        return Completed(value="done")

    monkeypatch.setattr("sys.stdin", io.StringIO("Hello from stdin\n"))
    monkeypatch.setattr(local_runner.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(local_runner, "run_prompt", _record_prompt)

    status = asyncio.run(run_cli([]))

    assert status == 0
    assert prompts == ["Hello from stdin"]


def _run_runtime_tool_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProviderStreamMock, io.StringIO]:
    """Run the local runner through a fake provider and real read tool."""

    provider = ProviderStreamMock(
        [
            tool_call_stream(
                response_id="resp_read",
                call_id="call_read",
                tool_name="read",
                arguments={"path": "notes.txt"},
            ),
            final_text_stream(
                response_id="resp_final",
                text="The note says hello.",
            ),
        ]
    )
    output = io.StringIO()
    (tmp_path / "notes.txt").write_text("hello from disk\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdout", output)

    asyncio.run(
        run_prompt(
            "Read the note",
            provider=provider,
        )
    )
    return provider, output


def _assert_runtime_event_sequence(output: io.StringIO) -> None:
    """Assert the local runner emitted the expected JSONL event sequence."""

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [line["type"] for line in lines] == [
        "run_start",
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
        "run_end",
    ]


def _assert_provider_received_tool_result(
    provider: ProviderStreamMock,
) -> None:
    """Assert the second provider call received the read tool result."""

    assert provider.await_count == 2
    initial_request_history = provider.history(0)
    assert len(initial_request_history) == 1
    assert expect_user_message(initial_request_history[0]).content == "Read the note"

    follow_up_request_history = provider.history(1)
    assert _expect_tool_text(follow_up_request_history[2]) == "hello from disk\n"


def _expect_tool_text(item: ConversationItem) -> str:
    """Assert and return the first text block from a tool result item."""

    tool_result = expect_tool_result_turn(item)
    content = tool_result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text

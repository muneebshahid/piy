"""Example local runner for one headless Tile prompt."""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from openai import AsyncOpenAI

from tile import (
    AgentRuntime,
    RunReport,
    SQLiteStore,
    Store,
)
from tile.events import AgentEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS
from tile.types import ToolDefinition
from examples.settings import settings


def main() -> None:
    """Run the example local runner."""

    raise SystemExit(asyncio.run(run_cli(sys.argv[1:])))


async def run_cli(argv: Sequence[str]) -> int:
    """Run a prompt from command arguments or standard input."""

    prompt = _read_prompt(argv, sys.stdin)
    if not prompt:
        print("Provide a prompt as arguments or stdin.", file=sys.stderr)
        return 2

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    report = await run_prompt(prompt, stream_fn=create_stream_api(client))
    return 0 if report.status == "completed" else 1


async def run_prompt(
    prompt: str,
    *,
    stream_fn: StreamFn,
    model: str | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    store: Store | None = None,
    cwd: Path | str | None = None,
    output: TextIO | None = None,
) -> RunReport:
    """Run one prompt through a runtime session and write JSON event lines."""

    active_tools = tuple(tools) if tools is not None else BUILTIN_TOOLS
    if store is None:
        owned_store: SQLiteStore | None = SQLiteStore(in_memory=True)
        active_store: Store = owned_store
    else:
        owned_store = None
        active_store = store
    try:
        runtime = AgentRuntime(
            stream_fn=stream_fn,
            model=model or settings.openai_model,
            store=active_store,
            tools=active_tools,
            cwd=cwd if cwd is not None else Path.cwd(),
        )
        session = runtime.session(name="local-runner")
        event_output = output or sys.stdout

        run = await session.prompt(prompt)
        async for event in run.events():
            event_output.write(_serialize_event(event))
            event_output.write("\n")
        return await run.wait()
    finally:
        if owned_store is not None:
            owned_store.close()


def _read_prompt(argv: Sequence[str], stdin: TextIO) -> str:
    """Read a prompt from positional arguments or standard input."""

    if argv:
        return " ".join(argv).strip()
    return stdin.read().strip()


def _serialize_event(event: AgentEvent) -> str:
    """Serialize one agent event for line-oriented local output."""

    return event.model_dump_json()


if __name__ == "__main__":
    main()

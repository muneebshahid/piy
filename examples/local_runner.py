"""Example local runner for one headless Tile prompt."""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from openai import AsyncOpenAI

from tile import (
    AgentHarness,
    Completed,
    Provider,
    RunResult,
    SessionRepository,
    SQLiteStore,
)
from tile.events import AgentEvent
from tile.providers.openai import OpenAIProvider
from tile.tools import BUILTIN_TOOLS
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
    provider = OpenAIProvider(client=client, model=settings.openai_model)
    result = await run_prompt(prompt, provider=provider)
    return 0 if isinstance(result, Completed) else 1


async def run_prompt(
    prompt: str,
    *,
    provider: Provider,
) -> RunResult:
    """Run one prompt through a session-bound harness and write JSON events."""

    store = SQLiteStore(in_memory=True)
    try:
        session = SessionRepository(store).create()
        harness = AgentHarness(
            session=session,
            tools=BUILTIN_TOOLS,
            cwd=Path.cwd(),
        )

        run = await harness.prompt(prompt, provider=provider)
        async for event in run.events():
            print(_serialize_event(event))
        return await run.wait()
    finally:
        store.close()


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

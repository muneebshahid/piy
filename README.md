# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Tile is a headless, persistent agent harness for Python applications. It handles
the model and tool loop, session history, run events, and persistence while your
application stays in control of the interface and deployment.

> [!WARNING]
> Tile is early-stage software. APIs may change without a deprecation period.
> OpenAI is the only supported provider for now, and Python 3.13 or newer is
> required.

## Install

```bash
uv add tile-runtime
```

The package is published as `tile-runtime` and imported as `tile`.

## Quickstart

Set `OPENAI_API_KEY`, then run:

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentHarness, Completed, SessionRepository, SQLiteStore
from tile.extensions import NonInteractive
from tile.providers.openai import OpenAIProvider
from tile.tools import BUILTIN_TOOLS


async def main() -> None:
    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create()
    harness = AgentHarness(
        session=session,
        instructions="You are a coding agent. Complete the requested task.",
        tools=BUILTIN_TOOLS,
        cwd=Path.cwd(),
        extensions=(NonInteractive(),),
    )
    provider = OpenAIProvider(
        client=AsyncOpenAI(),
        model="gpt-5.4",
    )

    run = await harness.prompt(
        "List the files in the current directory.",
        provider=provider,
    )
    result = await run.wait()
    if isinstance(result, Completed):
        print(result.value)

    store.close()


asyncio.run(main())
```

`instructions` is required because Tile does not impose an agent identity or
behavior. `NonInteractive` is explicit because not every agent must run without
requesting caller input.

`harness.prompt()` returns a handle for waiting on the result or consuming the
run's event stream:

```python
run = await harness.prompt("Inspect this repository", provider=provider)
async for event in run.events():
    print(event)

result = await run.wait()
```

Tile currently includes:

- persistent sessions and run records backed by SQLite;
- built-in file, search, edit, and shell tools;
- validated custom tools using Pydantic input models;
- typed results using your own Pydantic model;
- hooks and passive run-event observers;
- structured events for model output, tool calls, and run outcomes; and
- recovery from interrupted provider responses without committing partial
  conversation turns.

Hook results are not durable yet. `before_run` output is used for the live run
and enters history only when finalization commits. Tile cannot currently reopen
that run, restore the effective hook output, and skip the hook after a crash.
Persisting and consuming hook results during recovery will be introduced
together so the durability guarantee is complete.

See [the agent harness guide](docs/agent_harness.md) for the runtime model,
[the extensions guide](docs/extensions.md) for hooks and observers, and the
`examples` directory for a small local runner.

## Project status

Tile is still being shaped through use in real applications. Current work is
focused on a stable local runtime and durable sessions. Multi-provider support,
approval flows, persisted event replay, and service mode are planned but not
available yet.

## Security

The built-in tools are not sandboxed. The shell tool runs commands with the
same permissions as the Tile process, and file tools can access absolute paths.
Use a container or virtual machine when the agent should not have access to the
host system.

## Development

```bash
uv sync
make format
make type_check
make test
```

Run the example against the current directory:

```bash
uv run python -m examples.local_runner "Inspect this repository"
```

## License

[MIT](LICENSE)

# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python-native runtime for building your own agent harness.

The agents that actually work — the coding harnesses — share one
architecture: a frontier model, local tools, and your context, trusted to
finish a task. Tile is that architecture as an embeddable Python runtime.
Your model, your context, your software.

Tile is a **runtime you use as a library**: pass it a provider stream, tools,
and one store — built-ins ship for all three — and it runs prompt-driven agent
sessions on top: provider streaming, a tool-execution loop, typed run
outcomes, session history, and durable run summaries. Setup is one
constructor call; the quickstart below is the whole thing. Embed it in an
application, or build a service on top.

**Status: 0.x.** APIs change without deprecation cycles. OpenAI (Responses API)
is the only provider today; more are planned. Requires Python 3.13+.
See [Roadmap](#roadmap) for where this is going.

## Why a runtime?

Tile owns the lifecycle around an agent loop, and that ownership is a set of
concrete guarantees:

- a prompt becomes a task-owned `RunHandle`: `session.prompt(...)` returns it
  immediately, and execution continues even if every subscriber disconnects;
- a provider death never corrupts the session: partial turns are dropped,
  unanswered tool calls are healed, and the next prompt works;
- every accepted prompt has a durable running record before execution begins;
- every run log closes: exactly one `RunEndEvent` carries the terminal
  outcome on every in-process termination path;
- providers normalize into one event and history contract;
- prompts may require explicit, typed success or failure outcomes.

Tile does not provide graphs, teams, workflows, memory/RAG, a UI, or a
deployment platform. Applications compose those concerns around the runtime.

## Install

```bash
pip install tile-runtime
```

The distribution is `tile-runtime`; the import name is `tile`.

## Quickstart

With `OPENAI_API_KEY` set:

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentRuntime, SQLiteStore
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS


async def main() -> None:
    store = SQLiteStore(in_memory=True)
    runtime = AgentRuntime(
        stream_fn=create_stream_api(AsyncOpenAI()),
        model="gpt-5.4",
        tools=BUILTIN_TOOLS,
        cwd=Path.cwd(),
        store=store,
    )
    session = runtime.session(name="quickstart")
    run = await session.prompt("List the files in the current directory.")
    print(await run.wait())  # "completed"
    print(run.output_text)


asyncio.run(main())
```

`cwd` is required and is the runtime's single working directory: it is
announced to the model in the system prompt and injected into every tool whose
function declares a `cwd` parameter. `BUILTIN_TOOLS` (`read`, `bash`, `edit`,
`grep`, `find`, `ls`, `write`) are plain, unbound definitions — the runtime
binds them. Tool inputs are Pydantic models: Tile generates the provider schema
from the model and validates every model-supplied call before invocation.
A custom tool opts into the working directory the same way:

```python
from pathlib import Path

from pydantic import Field

from tile.types import ToolDefinition, ToolError, ToolInput, ToolResult


class SearchInput(ToolInput):
    query: str = Field(description="Text to search for.")


async def search(params: SearchInput, *, cwd: Path) -> ToolResult:
    if not params.query:
        raise ToolError("A search query is required.")
    ...  # cwd is injected and never exposed to the model


search_tool = ToolDefinition(
    name="search",
    description="Search the current workspace.",
    input_model=SearchInput,
    fn=search,
)
```

`ToolInput` rejects wrong types and extra fields. Tile passes the validated model
instance directly to the tool, preserving nested models, aliases, and defaults.
Validation errors are returned to the model for correction. Tool functions
return `ToolResult` only for success and raise `ToolError` for intentional,
model-visible failures. Any other exception is normalized as an unexpected
invocation failure, while cancellation continues to propagate.

Prompt execution is task-owned: `session.prompt(...)` submits a run and returns
a handle immediately, the runtime drives it to completion, and any number of
subscribers can observe the event stream.

```python
run = await session.prompt("Inspect the current repository")
async for event in run.events():
    ...
status = await run.wait()  # "completed" | "failed" | "aborted"
```

Every run's log begins with `RunStartEvent` and ends with exactly one
`RunEndEvent` carrying the run's terminal outcome, on every in-process
termination path. Inner events carry no such guarantee: a failure or
abort can tear the run down with inner scopes still open, and the run
end sweeps them — its outcome names why, exactly once. `run.wait()`
returns only after that closure, so waiters always observe a closed
log.

Run events are currently replayable in process while the `RunHandle` exists.
Conversation history and run records share one atomic SQLite store.
Cross-process event replay, approval resumption, and service mode are planned,
not current capabilities.

## Atomic persistence

One `Store` owns sessions, runs, and committed conversation history. A running
record contains the submitted prompt before provider execution begins. The
prompt and all replayable assistant/tool items remain provisional until the run
finishes; session history therefore contains complete committed turns only.

Execution sits between two short transactions:

1. `start_run` validates the session, snapshots committed history, and inserts
   the running record atomically.
2. Provider streaming and tool execution happen entirely in memory.
3. `finish_run` conditionally finalizes the still-running record and appends
   its complete history delta in one transaction.

```python
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentRuntime, SQLiteStore
from tile.providers.openai import create_stream_api


database_path = Path("tile.db")
store = SQLiteStore(database_path)
runtime = AgentRuntime(
    stream_fn=create_stream_api(AsyncOpenAI()),
    model="gpt-5.4",
    cwd=Path.cwd(),
    store=store,
)

session = runtime.session()
run = await session.prompt("Inspect this repository")
await run.wait()

record = runtime.get_run(run.id)
session_records = runtime.runs_for(session.id)
```

`finish_run` uses the run's `status="running"` condition as a stale-writer
fence. If another process has already replaced or finalized that run, no
history is inserted. If any terminal write fails, the transaction rolls back,
the stored run remains `running`, and `RunHandle.wait()` raises
`RunPersistenceError`. Recover explicitly by submitting another prompt with
`replace_active=True`.

```python
replacement = await session.prompt("Try again", replace_active=True)
```

Replacement atomically marks the still-running predecessor as
`Aborted(reason="replaced")` and inserts the successor. If the predecessor
finished before the transaction acquired its lock, its terminal result remains
unchanged and the new run starts normally.

Forking creates a new session and copies its complete committed history into
new history rows. Run records are not copied:

```python
fork = session.fork(session_id="experiment")
```

Custom `Store` implementations must provide the same atomic semantics. JSONL
or another append-only backend is valid only if it supplies locking,
all-or-nothing lifecycle records, recovery, and stale-writer fencing. A backend
that performs independent best-effort writes is not a valid persistent Store.
There is intentionally no migration from the earlier split-store development
schema; `SQLiteStore` rejects it with a clear schema error.

## Typed results

Pass a pydantic model to get a validated result object back instead of prose to
parse:

```python
from pydantic import BaseModel

from tile import AgentFailure, Completed, Failed


class WeatherReport(BaseModel):
    city: str
    temp_c: float
    summary: str


run = await session.prompt("What's the weather in Munich?", result=WeatherReport)
await run.wait()
match run.outcome:
    case Completed(value=report):
        print(report.city, report.temp_c)   # a WeatherReport instance
    case Failed(cause=AgentFailure(reason=reason)):
        print("model declared failure:", reason)
```

For that prompt only, the runtime registers a `complete` tool (whose schema is
your model) and a `fail(reason)` tool, and instructs the model to end the run
through one of them. Validation errors route back to the model as ordinary tool
errors for correction; a run that ends in plain text is reminded to deliver,
a bounded number of times. The names `complete` and `fail` are reserved —
caller tools may not use them.

**Designing result schemas:** demand judgment, not transcripts. The result
should be the model's *verdict* — small, typed fields it decides — not a
container for data your tools already produced (bulk data belongs on
`ToolResult.details`). Add a `summary: str` field when you want guaranteed
prose alongside the structure.

**Prompt caching:** reuse one `result=` schema per session. The result tools
and contract text sit at the front of every provider request, so alternating
typed and plain prompts — or switching schemas — within a session re-reads the
whole session history at full price on each flip.

## Status and outcome

`run.status` reflects the persistent record supplied when the handle starts or
finalizes. Use `runtime.get_run(run.id)` when an immediate cross-process
authoritative read is required. Every successfully persisted terminal run
carries exactly one `Completed`, `Failed`, or `Aborted` outcome, and status is
derived directly from that variant. A `Failed` outcome preserves whether the
model declined through `AgentFailure` or execution broke through
`ExecutionFailure`. `run.exception` retains an original in-process execution
exception for local debugging; it is never serialized.

| Run ending | `status` | `outcome` |
|---|---|---|
| Plain prompt, text answer | `completed` | `Completed(value=text)` |
| `complete` validates | `completed` | `Completed(value=model instance)` |
| `fail(reason)` | `failed` | `Failed(cause=AgentFailure(...))` |
| Reminder cap exhausted | `failed` | `Failed(cause=AgentFailure(...))` |
| Provider dies (stream error or raise) | `failed` | `Failed(cause=ExecutionFailure(...))` |
| Explicit cancellation | `aborted` | `Aborted(reason="cancelled")` |
| Replaced by a newer run | `aborted` | `Aborted(reason="replaced")` |
| Atomic finalization fails | remains `running` in the Store | live `Failed(cause=PersistenceFailure(...))`; `wait()` raises |

A provider death never corrupts the session: partial turns are dropped, history
ends at the last stable item, unanswered tool calls are healed, and the session
accepts the next prompt immediately. Tile does not retry; request-level retries
belong to the `AsyncOpenAI` client you construct (`max_retries`), and the
recovery unit above that is re-prompting the session.

Run events are replayable facts, and the run-level closure survives every
in-process termination: an exception or abort still lands exactly one
`RunEndEvent` as the log's final event before the terminal status lands.
After a successful finalization, `RunEndEvent.outcome`, `run.outcome`, and the
stored record agree. On persistence failure, the event carries a structured
persistence failure, `wait()` raises, and the stored run remains `running`.

## Observability

`run.events()` is the observation surface: every run yields a structured
event stream — run, agent, turn, message, and tool-execution scopes, plus
provider stream updates. Monitoring can rely on the closure guarantee: the
log begins with `RunStartEvent` and ends with exactly one `RunEndEvent`
whose outcome matches the run's terminal state, on every in-process
termination path. Failures are structured data, not log lines to parse: a
`Failed` outcome names its cause — the model's own `AgentFailure(reason=...)`
verdict, or an `ExecutionFailure` with an origin, exception type, and
message when a runtime boundary broke.

Planned, not current: one wide, high-cardinality telemetry record per run —
duration, token totals, per-tool aggregates, structured errors — delivered
to a caller-constructed sink. Tile core takes no telemetry-SDK dependency;
exporters and sampling remain application concerns.

## Testing your agent

`stream_fn` is a plain callable, so a scripted fake makes end-to-end tests
deterministic — no network, no API key:

```python
from tile.types import (
    ProviderSource,
    StreamDoneEvent,
    StreamStartEvent,
    TextBlock,
)

SOURCE = ProviderSource(provider="fake", model="fake-model")


async def fake_stream(history, model, *, instructions, tools):
    async def events():
        yield StreamStartEvent(source=SOURCE, response_id="resp_1")
        yield StreamDoneEvent(
            source=SOURCE,
            response_id="resp_1",
            stop_reason="stop",
            blocks=[TextBlock(text="All clear.")],
        )

    return events()


fake_stream.provider = "fake"
```

Hand `fake_stream` to `AgentRuntime` in place of the real provider and the
entire runtime executes: history is written, events are published, and the
run ends with a real outcome to assert on. Script a `tool_use` stop with a
`ToolCallBlock` to drive the tool loop, or a `complete` call to exercise a
typed result. A public `tile.testing` module with ready-made stream
builders is planned.

## Public API

Use the package facades for application code. Deep module paths are internal
and may move as Tile grows.

```python
from tile import (
    Aborted,
    AgentFailure,
    AgentRuntime,
    Completed,
    ExecutionFailure,
    Failed,
    HistoryItem,
    RunPersistenceError,
    RunHandle,
    RunRecord,
    SQLiteStore,
    Store,
)
from tile.events import AgentEvent, MessageEndEvent, RunEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS
from tile.types import ToolDefinition, ToolError, ToolInput, ToolResult
from tile.types import ToolInputValidationFailure, ToolInvocationFailure
```

`tile` exposes the runtime, session, run handle, persistent records, atomic
store, outcomes, and domain errors. `tile.events` exposes the structured
events yielded by `RunHandle.events()`. `tile.types` exposes provider-neutral
conversation, stream, and tool contracts, including structured validation and
invocation failures on tool-execution event details. `tile.providers.openai`
exposes
`create_stream_api`, which
binds a caller-constructed `AsyncOpenAI` client and optional provider reasoning
options to the runtime's stream-function contract:
`create_stream_api(AsyncOpenAI(...), reasoning={"effort": "medium"})`. A stream
function declares its provider identity via a `provider` attribute on the
callable, stated once where the callable is constructed.

## Architecture

```
tile/
├── providers/       # Provider integrations (OpenAI today)
├── store/           # Persistent records, Store contract, and SQLite adapter
├── tools/           # Built-in local tool implementations
├── types/           # Provider-neutral contracts for conversations and tools
├── agent.py         # Stateless agent loop: provider turns and tool batches
├── events.py        # Runtime event contracts and run lifecycle rules
├── prompt.py        # System prompt composition
├── result.py        # Typed run outcomes and the output-contract protocol
└── runtime/         # Session runtime package
    ├── handle.py    # RunHandle: live execution and event delivery
    ├── execution.py # Prompt programs: attempt loops and outcome derivation
    ├── history.py   # Provisional run-local conversation buffering
    ├── runtime.py   # AgentRuntime: orchestration and Store lifecycle
    └── session.py   # Session facade
tests/               # Test suite
```

## Roadmap

Development proceeds in validation-gated releases:

1. **Stable local runtime** (v0.1.0, shipped) — packaging, CI, typed results.
2. **Persistent sessions and run records** (current) — atomic run lifecycle,
   committed history, replacement fencing, and flat session forks.
3. **Multi-provider support** — hoist the normalized provider layer behind a
   conformance suite; Anthropic and ChatGPT-subscription providers.
4. **Downstream app validation** — a real application built on the embedded
   runtime decides what the runtime needs next.
5. **Proven runtime extensions and approval** — first hooks and a
   serializable pending-action state.
6. **Service mode and Python client** — `tile serve`: the same runtime behind
   a thin HTTP shell. Embed Tile as a library, or run it as a server.
7. **Durable service execution** — persisted run events, replay, worker
   leases, recovery.

## Security posture

Tile's built-in tools are deliberately unconfined. `bash` executes arbitrary
shell commands with the permissions of the process running the agent, and the
file tools accept absolute paths — the session working directory is a default,
not a sandbox. Run Tile only where you would run the model's commands yourself,
and use OS-level isolation such as a container or VM when you need a boundary.
Resource exhaustion from trusted local input is out of scope for now. Tool
authorization and first-class approval are planned, not current capabilities.

## Development

```bash
uv sync         # install dependencies
make test       # pytest
make format     # ruff
make type_check # ty
```

Run the example CLI against the current directory:

```bash
uv run python -m examples.local_runner "Inspect the current repository"
```

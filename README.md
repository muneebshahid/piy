# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python-native runtime for building your own agent harness.

The agents that actually work — the coding harnesses — share one
architecture: a frontier model, local tools, and your context, trusted to
finish a task. Tile is that architecture as an embeddable Python runtime.
Your model, your context, your software.

Tile is a **runtime you use as a library**. A `SessionRepository` manages
persistent sessions, an `AgentHarness` binds one session to its tools and
working directory, and a configured `Provider` is selected for each prompt.
Tile supplies provider streaming, a tool-execution loop, typed run outcomes,
session history, and durable run summaries. Embed it in an application, or
build a service on top.

**Status: 0.x.** APIs change without deprecation cycles. OpenAI (Responses API)
is the only provider today; more are planned. Requires Python 3.13+.
See [Roadmap](#roadmap) for where this is going.

## Why a runtime?

Tile owns the lifecycle around an agent loop, and that ownership is a set of
concrete guarantees:

- a prompt becomes a task-owned `RunHandle`: `harness.prompt(...)` returns it
  immediately, and execution continues even if every subscriber disconnects;
- a provider death never corrupts the session: partial turns are dropped,
  unanswered tool calls are healed, and the next prompt works;
- every accepted prompt has a durable running record before execution begins;
- every durable run log closes with `RunEndEvent`; a durability failure closes
  the local stream with `RunFaultEvent` and faults its harness;
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

from tile import AgentHarness, Completed, SessionRepository, SQLiteStore
from tile.providers.openai import OpenAIProvider
from tile.tools import BUILTIN_TOOLS


async def main() -> None:
    store = SQLiteStore(in_memory=True)
    session = SessionRepository(store).create()
    harness = AgentHarness(
        session=session,
        tools=BUILTIN_TOOLS,
        cwd=Path.cwd(),
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


asyncio.run(main())
```

`cwd` is required and is the harness's single working directory: it is
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

Prompt execution is task-owned: `harness.prompt(...)` submits a run and returns
a handle immediately, the harness drives it to completion, and any number of
subscribers can observe the event stream.

```python
run = await harness.prompt("Inspect the current repository", provider=provider)
async for event in run.events():
    ...
result = await run.wait()  # Completed | Failed | Aborted | Faulted
```

Every run's log begins with `RunStartEvent`. A durably finalized run ends with
exactly one `RunEndEvent` carrying its outcome. If finalization itself fails,
the local log instead ends with `RunFaultEvent` and `run.wait()` returns
`Faulted`. Inner events carry no closure guarantee: a failure or abort can
leave inner scopes open, and the terminal run event closes them.

Run events are currently replayable in process while the `RunHandle` exists.
Conversation history and run records share one atomic SQLite store.
Cross-process event replay, approval resumption, and service mode are planned,
not current capabilities.

## Atomic persistence

One `Store` owns sessions, runs, and committed conversation history. Persistent
`SessionRecord` and `RunRecord` objects are outputs of Store operations, not
caller-constructed inputs. A running record contains the submitted prompt
before provider execution begins. The prompt and all replayable assistant/tool
items remain provisional until the run finishes; session history therefore
contains complete committed turns only.

Prompt admission supplies intent fields — the owning session id, a new run id,
the prompt, model, and provider — and the Store constructs and persists the
running record. Finalization supplies the owning session id, run id, outcome,
and history delta. The Store scopes the operation by both identities and
derives the authoritative terminal record from its stored running row. Reading
one run is scoped by session in the same way. `RunRecord` consequently has no
public `start` or `finish` factory: it is an immutable snapshot returned after
a Store operation succeeds.

Execution sits between two short transactions:

1. `start_run` validates the session, snapshots committed history, constructs
   the running record from the supplied intent fields, and inserts it
   atomically.
2. Provider streaming and tool execution happen entirely in memory.
3. Execution derives a candidate outcome. `finish_run` finds the authoritative
   stored record by session and run id, applies the outcome, and persists the
   complete history delta in one transaction before exposing `RunEndEvent`.

```python
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentHarness, SessionRepository, SQLiteStore
from tile.providers.openai import OpenAIProvider


database_path = Path("tile.db")
store = SQLiteStore(database_path)
repository = SessionRepository(store)
session = repository.create()
harness = AgentHarness(
    session=session,
    cwd=Path.cwd(),
)
provider = OpenAIProvider(client=AsyncOpenAI(), model="gpt-5.4")

run = await harness.prompt("Inspect this repository", provider=provider)
await run.wait()

record = next(record for record in session.get_runs() if record.id == run.id)
session_records = session.get_runs()
```

`finish_run` uses the run's `status="running"` condition as an ended-run fence.
If the escape hatch already aborted the record, its handle returns the
authoritative stored outcome. If the terminal write fails because of a backend
error, the transaction rolls back, the stored run remains `running`, and
`RunHandle.wait()` returns `Faulted`. The running record blocks later prompts
with `ActiveRunError`.

```python
aborted = repository.abort_active_run(session.id)
assert aborted is not None

retry = await harness.prompt(
    "Try again",
    provider=provider,
)
```

`abort_active_run` is a temporary escape hatch for clearing a durable record
after its persistence problem has been resolved. It atomically records
`Aborted(reason="cancelled")` and returns `None` when no run is active. It is not
process control: an old task may continue, but ended-run fencing prevents it
from committing. Normal callers should use `RunHandle.abort()` when the local
handle is available.

Forking creates a new session and copies its complete committed history into
new history rows. Run records are not copied:

```python
fork = repository.fork(session.id, target_session_id="experiment")
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


run = await harness.prompt(
    "What's the weather in Munich?",
    provider=provider,
    result=WeatherReport,
)
result = await run.wait()
match result:
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

## Run results and durable records

`await run.wait()` returns only a `RunResult`: `Completed`, `Failed`,
`Aborted`, or `Faulted`. It does not bundle persistence metadata or history.
Read committed history and durable run records from the harness's `Session`:
`session.get_history()` and `session.get_runs()`.

A `Failed` result preserves whether the agent could not satisfy the result
contract through `AgentFailure` or execution broke through
`ExecutionFailure`. `Faulted` is deliberately not a persisted `RunOutcome`;
it means the harness could not durably establish the terminal outcome.

| Run ending | `RunResult` |
|---|---|
| Plain prompt, text answer | `Completed(value=text)` |
| `complete` validates | `Completed(value=model instance)` |
| `fail(reason)` or reminder cap | `Failed(cause=AgentFailure(...))` |
| Provider dies | `Failed(cause=ExecutionFailure(...))` |
| Local or durable abort | `Aborted(reason="cancelled")` |
| Atomic finalization fails | `Faulted(error=StorePersistenceError(...))` |

A provider death never corrupts the session: partial turns are dropped, history
ends at the last stable item, unanswered tool calls are healed, and the session
accepts the next prompt immediately. Tile does not retry; request-level retries
belong to the `AsyncOpenAI` client you construct (`max_retries`), and the
recovery unit above that is re-prompting the session.

Run events are replayable in-process facts. After successful finalization,
`RunEndEvent.outcome`, `run.wait()`, and the stored record agree. A backend
persistence failure instead emits `RunFaultEvent`, returns `Faulted`, and may
leave the stored run `running` for explicit recovery.

## Observability

`run.events()` is the observation surface: every run yields a structured
event stream — run, agent, turn, message, and tool-execution scopes, plus
provider stream updates. Monitoring can rely on the closure guarantee: the
log begins with `RunStartEvent` and ends with `RunEndEvent` after durable
finalization or `RunFaultEvent` after a durability failure. Failures are
structured data, not log lines to parse: a
`Failed` outcome names its cause — the model's own `AgentFailure(reason=...)`
verdict, or an `ExecutionFailure` with an origin, exception type, and
message when a runtime boundary broke.

Planned, not current: one wide, high-cardinality telemetry record per run —
duration, token totals, per-tool aggregates, structured errors — delivered
to a caller-constructed sink. Tile core takes no telemetry-SDK dependency;
exporters and sampling remain application concerns.

## Testing your agent

`Provider` is an abstract callable, so a scripted implementation makes
end-to-end tests deterministic — no network, no API key:

```python
from tile import Provider
from tile.types import (
    ProviderSource,
    StreamDoneEvent,
    StreamStartEvent,
    TextBlock,
)

SOURCE = ProviderSource(provider="fake", model="fake-model")


class FakeProvider(Provider):
    def __init__(self):
        super().__init__(model="fake-model")

    @property
    def name(self):
        return "fake"

    async def stream(self, history, *, instructions, tools):
        async def events():
            yield StreamStartEvent(source=SOURCE, response_id="resp_1")
            yield StreamDoneEvent(
                source=SOURCE,
                response_id="resp_1",
                stop_reason="stop",
                blocks=[TextBlock(text="All clear.")],
            )

        return events()
```

Pass `FakeProvider()` to `harness.prompt`. The entire harness executes:
history is written, events are emitted, and the run ends with a real outcome
to assert on. Script a `tool_use` stop with a
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
    AgentHarness,
    Completed,
    ExecutionFailure,
    Failed,
    Faulted,
    HistoryItem,
    Provider,
    RunAlreadyEndedError,
    RunHandle,
    RunRecord,
    Session,
    SessionRepository,
    SQLiteStore,
    Store,
    StorePersistenceError,
)
from tile.events import AgentEvent, MessageEndEvent, RunEndEvent, RunFaultEvent
from tile.providers.openai import OpenAIProvider
from tile.tools import BUILTIN_TOOLS
from tile.types import ToolDefinition, ToolError, ToolInput, ToolResult
from tile.types import ToolInputValidationFailure, ToolInvocationFailure
```

`tile` exposes the harness, session repository, session, provider, run handle,
persistent records, atomic store, results, and domain errors. `tile.events`
exposes the structured
events yielded by `RunHandle.events()`. `tile.types` exposes provider-neutral
conversation, stream, and tool contracts, including structured validation and
invocation failures on tool-execution event details. `tile.providers.openai`
exposes `OpenAIProvider`, which binds a caller-constructed `AsyncOpenAI`
client, model, and optional reasoning options:
`OpenAIProvider(client=AsyncOpenAI(...), model="gpt-5.4",
reasoning={"effort": "medium"})`.

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
├── sessions.py      # Session and SessionRepository persistence boundary
└── runtime/         # Single-session harness package
    ├── harness.py     # AgentHarness: session-bound prompt coordination
    ├── run_execution.py # RunExecution: live and durable run ownership
    ├── run_handle.py  # RunHandle: caller-facing live-run facade
    ├── execution.py   # Prompt programs: attempt loops and outcome derivation
    └── history.py     # Provisional run-local conversation buffering
tests/               # Test suite
```

## Roadmap

Development proceeds in validation-gated releases:

1. **Stable local runtime** (v0.1.0, shipped) — packaging, CI, typed results.
2. **Persistent sessions and run records** (current) — atomic run lifecycle,
   committed history, ended-run fencing, and flat session forks.
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

# Extensions

Extensions package optional behavior for an `AgentHarness`. They can register
lifecycle hooks that influence a run and observers that react to its events.

Tile currently supports:

- the `before_run` hook;
- asynchronous run-event stream observers;
- the built-in `NonInteractive` and `EventLogger` extensions.

An extension is any object with a `register()` method. It does not need to
inherit from a Tile base class.

```python
from tile import AgentHarness
from tile.extensions import (
    BeforeRunContext,
    BeforeRunResult,
    ExtensionRegistry,
    RunEventStream,
)


class MyExtension:
    """Contribute optional behavior to one harness."""

    def register(self, registry: ExtensionRegistry) -> None:
        registry.before_run(self.before_run)
        registry.observe(self.observe)

    async def before_run(
        self,
        context: BeforeRunContext,
    ) -> BeforeRunResult:
        return BeforeRunResult(
            system_prompt=f"{context.system_prompt}\n\nNever expose secrets.",
        )

    async def observe(self, stream: RunEventStream) -> None:
        async for event in stream:
            print(event.type)


harness = AgentHarness(
    session=session,
    cwd=cwd,
    instructions=instructions,
    extensions=(MyExtension(),),
)
```

`AgentHarness` calls `register()` once during construction. The registered hook
then runs once for each prompt. The observer is also called once per run and
receives an asynchronous stream containing that run's events.

## Extension boundaries

Extensions deliberately have a small surface:

| Capability | Configuration |
| --- | --- |
| Lifecycle changes | Register a hook through `ExtensionRegistry` |
| Passive event handling | Register an observer through `ExtensionRegistry` |
| Model-callable tools | Pass them directly to `AgentHarness(tools=...)` |
| Persistence | Supply a `Store` to `SessionRepository` |
| Typed result contracts | Pass `result_type` to `AgentHarness.prompt()` |

The registry does not accept tools, persistence implementations, result
strategies, or custom event types. These capabilities have different lifecycle
and failure rules and remain explicit on their owning APIs.

## Registration and ordering

Every extension receives the same `ExtensionRegistry`, in the order supplied to
`AgentHarness`. Contributions are also retained in registration order.

```python
class ExtensionRegistry:
    def before_run(self, hook: BeforeRunHook) -> None: ...

    def observe(self, observer: RunObserver) -> None: ...
```

After registration, the harness builds immutable collections of hooks and
observers. The registry itself is not passed into a run or callback.

## The `before_run` hook

`before_run` can replace the system prompt and append messages to the input for
one run.

```python
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from tile.types import ConversationItem


class BeforeRunContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    run_id: str
    system_prompt: str
    messages: tuple[ConversationItem, ...]


class BeforeRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = None
    additional_messages: tuple[ConversationItem, ...] = ()


type BeforeRunHook = Callable[
    [BeforeRunContext],
    Awaitable[BeforeRunResult | None],
]
```

The hook receives:

- the allocated session and run IDs;
- the complete system prompt for this run;
- the run's initial messages, beginning with the caller's user message.

`messages` does not contain the session's existing committed history. It only
contains input being added by the current run. Existing history is joined with
these messages when execution starts.

The system prompt is compiled before hooks run. When `result_type` is used, it
already contains Tile's result-contract instructions. An extension that returns
`system_prompt` replaces that complete value, so it should retain any earlier
instructions it still needs:

```python
from tile.extensions import BeforeRunContext, BeforeRunResult, ExtensionRegistry


class AddPolicy:
    def register(self, registry: ExtensionRegistry) -> None:
        registry.before_run(self.before_run)

    async def before_run(
        self,
        context: BeforeRunContext,
    ) -> BeforeRunResult:
        return BeforeRunResult(
            system_prompt=f"{context.system_prompt}\n\nNever expose secrets.",
        )
```

Returning `None` leaves the context unchanged. `additional_messages` are
appended; they do not replace existing messages.

### Composition

Hooks run sequentially in registration order. Each hook sees the context
produced by the preceding hook:

```text
caller input
-> first before_run hook
-> validate and apply its result
-> second before_run hook
-> validate and apply its result
-> admit and execute the run
```

This makes ordering significant. Extensions that both replace the system prompt
must compose it intentionally.

### Validation

Each hook receives a defensive copy of its context. Tile validates every returned
`BeforeRunResult` before applying it.

Additional messages must form valid conversation input. In particular:

- tool-call IDs must be unique;
- a tool result must reference a preceding call and use the same tool name;
- a call can have only one result;
- every tool call added by a hook must have a result in that hook's output.

Invalid output raises before run admission. Tile relies on static typing for the
callable signature and does not inspect hook parameters at registration time.
Normal Python invocation or awaiting still exposes invalid callable shapes at
runtime.

### Failure behavior

`before_run` is part of the run's control path. If it raises, or returns invalid
output:

- `AgentHarness.prompt()` raises the error;
- no `RunHandle` is returned;
- no run record is created;
- the provider is not invoked.

The harness remains reusable after the problem is resolved.

### Lifecycle and persistence

The current flow is:

```text
AgentHarness.prompt
-> allocate run ID
-> build the complete system prompt
-> run before_run hooks
-> persist the active run record with the caller prompt
-> start provider execution with the hook-resolved input
-> finalize the outcome and complete conversation history atomically
```

Hook-resolved messages remain provisional together with generated assistant and
tool messages. They enter committed session history only when finalization
commits successfully, regardless of whether the run outcome is `Completed`,
`Failed`, or `Aborted`. The hook-resolved system prompt is not stored in
`RunRecord`.

Durable hook results are planned but not supported yet. Tile does not currently
resume an existing run, reload a previous `before_run` result, or skip a hook
because it ran before a crash. Supporting that requires one complete recovery
contract: persist the effective hook output, reopen the same run, restore that
output, and continue without invoking the hook again. Persisting the output
without consuming it during recovery would provide misleading durability, so it
is intentionally deferred.

## Observers

Observers passively consume one run's event stream:

```python
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from tile.events import AgentEvent


@dataclass(frozen=True)
class RunEventStream:
    session_id: str
    run_id: str
    events: AsyncIterator[AgentEvent]

    def __aiter__(self) -> AsyncIterator[AgentEvent]: ...


type RunObserver = Callable[[RunEventStream], Awaitable[None]]
```

Register an observer with `registry.observe()`:

```python
from tile.extensions import ExtensionRegistry, RunEventStream


class EventCounter:
    def __init__(self) -> None:
        self.count = 0

    def register(self, registry: ExtensionRegistry) -> None:
        registry.observe(self.observe)

    async def observe(self, stream: RunEventStream) -> None:
        async for _ in stream:
            self.count += 1
```

Observers receive run lifecycle events and the agent events emitted between
them. Delivery has these rules:

- each observer is invoked once per run in its own task;
- every observer receives an independent iterator beginning with `run_start`;
- events are yielded in publication order through `run_end` or `run_fault`;
- each iterator yields defensive event copies, so a consumer cannot modify the
  runtime log or another consumer's values;
- an ordinary observer exception is logged, ends that observer for the current
  run, and does not interrupt other observers;
- cancelling an observer task stops only that observer, not the run;
- observer failures never change the run result.

`RunHandle.wait()` waits for execution and durable finalization, not observer
completion. An observer that is temporarily behind can continue consuming the
buffered stream after `wait()` returns. Observer delivery remains best effort:
process exit can discard unfinished observers, and a slow or stuck observer
retains its task and the run's in-memory event log. Observers are therefore not
a durability mechanism.

Observers do not run inside event publication and asynchronous I/O does not
delay `_emit()`. They still share the application's event loop: CPU-heavy work,
blocking I/O, or long synchronous logging handlers inside an observer can block
other tasks and should be offloaded.

Applications can also consume `RunHandle.events()` directly. That pull-based
stream and registered observers use the same in-memory event log. Every call
gets an independent cursor, so the caller and all observers can consume the
complete sequence at their own pace.

## Built-in extensions

### `NonInteractive`

`NonInteractive` prepends instructions telling the agent to work without waiting
for caller input. Tile does not enable it by default.

```python
from tile.extensions import NonInteractive

harness = AgentHarness(
    session=session,
    cwd=cwd,
    instructions=instructions,
    extensions=(NonInteractive(),),
)
```

Because it uses `before_run`, its instructions are applied independently to each
prompt.

### `EventLogger`

`EventLogger` logs every observed event with its session ID and run ID.

```python
import logging

from tile.extensions import EventLogger

harness = AgentHarness(
    session=session,
    cwd=cwd,
    instructions=instructions,
    extensions=(
        EventLogger(logging.getLogger("my_agent.events"), level=logging.DEBUG),
    ),
)
```

## Implementation map

The extension package separates public contracts from runtime orchestration:

```text
tile/extensions/
├── __init__.py              # public extension API
├── registry.py              # registration and harness assembly
├── non_interactive.py       # built-in before_run extension
├── event_logger.py          # built-in observer extension
├── run_observers.py         # stream contract and failure-isolated observer tasks
└── hooks/
    ├── __init__.py          # public hook exports
    ├── before_run.py        # context, result, validation, and application
    └── run_hooks.py         # ordered run-scoped hook orchestration
```

`AgentHarness` performs registration and retains the assembled `RunHooks` and
`RunObservers`. `RunExecution` owns the event log and starts one independent
stream consumer for every registered observer. Provider execution does not know
which extension supplied an instruction, message, or observer.

New hook points are added as explicit typed contracts rather than string names
in a generic dispatcher. Until another registry method is implemented and
exported, that hook point is not part of Tile's extension API.

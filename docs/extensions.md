# Extensions

Tile has three customization mechanisms:

- **Strategies** implement a core capability. One strategy is selected explicitly
  for each capability. Persistence uses `Store`; compaction is planned. Result
  contracts remain in core unless Tile gains another contract-enforcement mode.
- **Hooks** are awaited lifecycle callbacks. They receive immutable context and
  return a typed decision. The runtime applies the decision.
- **Observers** consume events without affecting execution. Their failures do not
  affect the run.

An extension packages tools, hooks, observers, or strategies. Registering an
extension may add tools, hooks, and observers. Strategies are still selected
explicitly by the caller.

## Registration

Extensions register contributions through a closed API:

```python
class MyExtension:
    """Contribute one harness feature."""

    def register(self, registry: ExtensionRegistry) -> None:
        registry.add_tools(self.tools)
        registry.before_run(self.before_run)
        registry.observe(self.observer)


harness = AgentHarness(
    session=session,
    cwd=cwd,
    extensions=[MyExtension()],
)
```

Direct harness tools and extension tools form one catalog. Duplicate names are
rejected.

### Assembly

`ExtensionRegistry` is a mutable builder used only while constructing the harness.
`AgentHarness` calls each extension's `register()` once, then freezes the collected
handlers into `RunHooks`.

```python
class ExtensionRegistry:
    """Collect extension contributions during harness construction."""

    def __init__(self) -> None:
        self._tools: list[ToolDefinition] = []
        self._before_run: list[BeforeRunHook] = []
        self._observers: list[Observer] = []

    def add_tools(self, tools: Sequence[ToolDefinition]) -> None:
        self._tools.extend(tools)

    def before_run(self, handler: BeforeRunHook) -> None:
        self._before_run.append(handler)

    def observe(self, observer: Observer) -> None:
        self._observers.append(observer)

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools)

    @property
    def observers(self) -> tuple[Observer, ...]:
        return tuple(self._observers)

    def build_run_hooks(self) -> RunHooks:
        return RunHooks(before_run=self._before_run)


def _register_extensions(
    extensions: Sequence[Extension],
) -> ExtensionRegistry:
    registry = ExtensionRegistry()
    for extension in extensions:
        extension.register(registry)
    return registry
```

```python
class AgentHarness:
    def __init__(
        self,
        *,
        extensions: Sequence[Extension] = (),
        ...,
    ) -> None:
        registry = _register_extensions(extensions)
        self._run_hooks = registry.build_run_hooks()
        self._tool_executor = ToolExecutor((*tools, *registry.tools))
        self._observers = registry.observers

    async def prompt(self, ...) -> RunHandle:
        execution = await RunExecution.start(
            ...,
            hooks=self._run_hooks,
        )
        return RunHandle(execution)
```

The registry is not passed to a run or handler. It stores bound handler methods;
`RunHooks` later invokes those methods with hook context.

## Hooks

Hooks are a closed catalog. Each hook defines its own context, result, composition,
durability, replay, and failure behavior.

`RunExecution` owns run lifecycle hooks. A typed `RunHooks` object keeps their
registered handlers and exposes one method per hook; there is no generic `Hook`
base class, string-keyed hook dictionary, or runtime signature inspection.

The first version contains only `before_run`. Each future hook adds an explicit
registry method, handler collection, constructor argument, and `RunHooks` method.
For example, `before_run_end` adds `_before_run_end` and
`RunHooks.before_run_end()` rather than going through a generic executor.

Hook failures are not ignored. A failure before admission propagates to the caller
without creating a run. A failure after admission fails that run according to the
hook's contract.

### `before_run`

`before_run` runs once in `RunExecution.start()`, before persistence:

```python
class BeforeRunContext(BaseModel):
    """Current run input visible to one handler."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    run_id: str
    system_prompt: str
    messages: tuple[ConversationItem, ...]


class BeforeRunResult(BaseModel):
    """Changes requested by one handler."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = None
    additional_messages: tuple[ConversationItem, ...] = ()


type BeforeRunHook = Callable[
    [BeforeRunContext],
    Awaitable[BeforeRunResult | None],
]


class RunHooks:
    """Run lifecycle hook handlers."""

    def __init__(self, *, before_run: Sequence[BeforeRunHook] = ()) -> None:
        self._before_run = tuple(before_run)

    async def before_run(
        self,
        context: BeforeRunContext,
    ) -> BeforeRunContext:
        current = context
        for handler in self._before_run:
            result = await handler(current.model_copy(deep=True))
            current = _apply_before_run_result(current, result)
        return current


def _apply_before_run_result(
    context: BeforeRunContext,
    result: BeforeRunResult | None,
) -> BeforeRunContext:
    """Apply one handler decision to the context for the next handler."""

    if result is None:
        return context
    system_prompt = (
        context.system_prompt if result.system_prompt is None else result.system_prompt
    )
    return BeforeRunContext(
        session_id=context.session_id,
        run_id=context.run_id,
        system_prompt=system_prompt,
        messages=(*context.messages, *result.additional_messages),
    )
```

Handlers run in registration order. Each handler sees the system prompt and
messages produced by previous handlers. A returned `system_prompt` replaces the
current value; `additional_messages` are appended.

`messages` contains input for this run, not the full session history. It initially
contains the caller prompt. Extensions may append any valid `ConversationItem`,
including assistant and tool-result messages. Tile validates the resulting
conversation before admission.

The system prompt is fully compiled before the hook runs. A trusted extension may
replace core or earlier-extension instructions.

```text
RunExecution.start
-> allocate run ID
-> compile the complete system prompt
-> create BeforeRunContext with the caller prompt
-> run before_run handlers
-> validate the final context
-> atomically persist the run, system prompt, and initial messages
-> emit RunStartEvent
-> begin execution
```

`RunExecution.start()` is async. It stores `RunHooks` for later lifecycle points.
`AgentHarness.prompt()` only supplies configuration and awaits
`RunExecution.start()`.

`RunExecution` carries the complete `RunHooks` object for the lifetime of the run.
It invokes run-boundary hooks itself and passes the same object down to the layer
that owns more specific boundaries:

```text
RunExecution.start       -> hooks.before_run
execute_prompt finish    -> hooks.before_run_end
tool execution boundary  -> hooks.before_tool / hooks.after_tool
```

This keeps every hook scoped to the run without moving tool or provider behavior
into `RunExecution`.

Initial messages are persisted during admission. `_RunHistory` therefore starts
from the admitted history instead of creating `UserMessage(prompt)` itself.

### `before_run_end`

`before_run_end` runs when the run would normally finish: no tool continuation or
other accepted work remains. It receives the messages belonging to the current
run, including agent messages. It may return a follow-up for the runtime to append
and continue within the same run.

It does not receive a completed `RunOutcome`, mutate history, or start another run.
Its exact multi-handler composition is still open.

## Result contracts

Result contracts remain a core mode selected with
`AgentHarness.prompt(..., result_type=Model)`. Core owns their instructions,
`complete` and `fail` tools, validation, follow-ups, retry limit, events, and final
outcome.

## Observers

Observers receive a read-only run event stream with `session_id` and `run_id`.
They run independently and off the execution path. Delivery is best effort across
process death.

Telemetry is an observer and owns its own sink. Tile does not pass its runtime
`Store` to telemetry. Durable data required for resume belongs to the core journal,
not an observer.

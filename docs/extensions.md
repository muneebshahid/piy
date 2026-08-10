# Extensions

Tile exposes three mechanically distinct ways to customize the harness:

- **Strategies** are explicit dependencies that implement a core capability. There can be
only one strategy can be selected per capability. The `Store`, for example, is selected when constructing `SessionRepository`.
  Current strategies include
  - Persistence (Implemented via `Store`)
  - Compaction (Planned)
  - Contract Enforcement (Part of core atm, in case of a need for multiple strategies, this will be extracted)
- **Hooks** are awaited callbacks at named points in the harness lifecycle. Each
  hook receives immutable input and may return only that hook's typed decision.
  The core validates and applies the decision; hooks never mutate harness internals
  directly.
- **Observers** passively consume run events. They cannot affect execution, and
  their failures do not affect the harness or run.

An **extension** packages one or more of these contributions together.

## Registration

Extensions register their contributions explicitly.

```python
class MyExtension:
    """Contribute one self-contained harness feature."""

    name = "my_extension"

    def register(self, registry: ExtensionRegistry) -> None:
        registry.observe(self.observer)
        registry.before_run(self.before_run)

harness = AgentHarness(
    session=session,
    cwd=cwd,
    extensions=[MyExtension()],
)
```

## Hooks

Hooks are a closed catalog of named decision points. For every hook Tile defines:

- its immutable input;
- its typed result;
- how multiple results compose;
- when its decision becomes durable;
- whether and when it is replayed;

Handlers run in registration order. Handler failures abort the run and raise an error to the caller.

### Hook Executor

### `before_run`

```python
class Hook(ABC):

    def __init__(self, func: Callable[[BaseModel | None], BaseModel | None], context_type: BaseModel | None, return_type: BaseModel | None):
        self.context_type = context_type
        self.return_type = return_type
        self.func = func
        self._verify_signature()

    async def __call__(self, context: BaseModel | None) -> BaseModel | None:
        # do processing and validate the result
        result = await self.func(context)
        self._validate(result)
        return result

    def _verify_signature(self) -> None:
        # verify that func signature matches context_type and return_type
        ...

    def _validate(self, result: BaseModel | None) -> None:
        # validate result is instance of return_type
        ...

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hook) and self.func is other.func

class BeforeRunContext:
    """Immutable input to the `before_run` hook."""

    session: Session
    run_id: str
    system_prompt: str
    messages: list[ConversationItem]

class BeforeRunResult:
    """Typed result of the `before_run` hook."""

    modified_system_prompt: str | None
    additional_messages: list[ConversationItem] | None

class BeforeRunHook(Hook):
    """Validate and apply the final preparation for the run."""

    def __init__(self, func: Callable[[BeforeRunContext], BeforeRunResult]):
        super().__init__(context_type=BeforeRunContext, return_type=BeforeRunResult, func=func)

    def _validate(self, result: BeforeRunResult) -> None:
        # verify that toolcalls if any also have responses etc

# Or may be there is a more pythonic way e.g. BeforeRunHook = Hook[BeforeRunContext, BeforeRunResult]()

class ExtensionRegistry:
    _observers: list[Observer] = []
    _before_run_hooks: list[BeforeRunHook] = []

    def observe(self, func: Callable[AsyncIterator[AgentEvents], Awaitable[None]]) -> None:
        """deduplicate and append observer, same observer instance may not be registered twice"""
        observer = Observer(func)
        if observer not in self._observers:
            self._observers.append(observer)

    def before_run(self, func: Callable[[BeforeRunContext], BeforeRunResult]) -> None:
      """deduplicate and append before_run hook"""
        hook = BeforeRunHook(func)
        if hook not in self._before_run_hooks:
            self._before_run_hooks.append(hook)

    def hooks(self) -> dict[str, list[Hook]]:
        return {
            "before_run": self._before_run_hooks
        }

class BeforeRunContext:
    """Immutable input to the `before_run` hook."""

    session: Session
    run_id: str
    system_prompt: str
    messages: list[ConversationItem]

    def deepcopy(self) -> Self:
      pass

class BeforeRunResult:
    """Typed result of the `before_run` hook."""

    modified_system_prompt: str | None
    additional_messages: list[ConversationItem] | None

class HookExecutor:
    """Run all registered handlers for a given hook."""

    def __init__(self, hooks: dict[str, list[Hook]]):
        self._hooks = hooks

    def before_run(self, context: BeforeRunContext):
        for hook in self._hooks["before_run"]:
            result = hook(context.deepcopy())
            if result.modified_system_prompt:
                context.system_prompt = result.modified_system_prompt
            if result.additional_messages:
                context.additional_messages.extend(result.additional_messages)


            # compose results, validate, apply, etc.

```

```text
generate run id
-> compile the complete system prompt for the run
-> run before_run handlers in registration order
-> validate and apply the final preparation
-> persist the effective run intent
-> emit RunStartEvent
-> begin provider execution
```

### `before_run_end`

This is a possible future hook at the normal would-finish boundary: no tool
continuation or other accepted work remains. It receives run facts, not a completed
`RunOutcome`, because the run has not ended yet. It may request a follow-up message;
the core durably appends that message and continues the same bounded run loop.

The exact multi-handler composition and error policy for this hook remain open and
must be decided before it is implemented.

## Observers

Observers receive an independent, read-only view of one run. A run observation
should include correlation metadata such as `session_id` and `run_id`, together
with an event stream.

```python
class MyObserver:
    """Observe one run without influencing it."""

    async def listen(self, observation: RunObservation) -> None:
        async for event in observation.events():
            ...
```

Each observer runs independently and off the execution path. If it raises, that
observer stops or reports its own error; execution continues. Observer delivery is
best effort across process death.

Telemetry is an observer and owns its own sink, database, or client. Tile does not
pass its runtime `Store` into telemetry. If a log is required for checkpointing or
resume, it is a core-owned durable operation journal rather than an observer.

A test recorder is a suitable first observer. A public logger must be safe by
default and avoid logging prompts, reasoning, tool arguments, tool results, or file
contents unless content capture is explicitly enabled.

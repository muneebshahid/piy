# Agent harness

Tile separates persistent session access from live prompt execution.

```python
store = SQLiteStore("tile.db")
repository = SessionRepository(store)
session = repository.create(name="work")
harness = AgentHarness(session=session, cwd=Path.cwd(), tools=tools)

provider = OpenAIProvider(
    client=client,
    model="gpt-5.4",
    reasoning={"effort": "high"},
)
run = await harness.prompt("Inspect the repository", provider=provider)
result = await run.wait()
```

## Responsibilities

- `SessionRepository` creates, gets, lists, forks, and deletes persistent
  sessions. It temporarily exposes a durable active-run abort escape hatch.
- `Session` is a lightweight, Store-bound handle. It exposes current metadata,
  committed history, and durable run records.
- `AgentHarness` owns the tools, working directory, instructions, and prompt
  admission configuration for exactly one session. It does not retain runs.
- `Provider` is the configured streaming abstraction used by execution.
  Implementations own their model, reasoning configuration, and transport. A
  caller may choose a different provider for each prompt.
- `OpenAIProvider` implements `Provider` with the OpenAI Responses API.
- `RunExecution` privately owns durable run admission, execution, event
  delivery, local cancellation, and finalization.
- `RunHandle` is the small caller-facing facade over a live execution.

## Durability contract

Prompt admission persists the running `RunRecord` before a handle is returned
or provider work begins. A start-write failure raises to the caller, returns no
handle, and faults the harness.

Successful execution is not exposed as terminal until its outcome and
replayable history commit atomically. `RunHandle.wait()` then returns only the
`Completed`, `Failed`, or `Aborted` outcome, and the event stream closes with a
matching `RunEndEvent`.

If terminal persistence fails, `wait()` returns `Faulted` and the stream closes
with `RunFaultEvent`. `Faulted` is not a persisted `RunOutcome`; the durable run
may still be `running`, so the Store rejects another prompt with
`ActiveRunError`. After fixing the persistence issue, callers may use the
temporary `SessionRepository.abort_active_run(session.id)` escape hatch. The
same harness can then be reused because it retains no fault state.

The escape hatch records `Aborted(reason="cancelled")`. It does not control a
local task, but late finalization returns the authoritative stored outcome.
Normal callers should use `RunHandle.abort()` for process-local cancellation.

## Result and state access

`await run.wait()` intentionally returns no report wrapper. Read orthogonal
state from the object that owns it:

```python
result = await run.wait()
history = session.get_history()
runs = session.get_runs()
record = next(record for record in runs if record.id == run.id)
```

This keeps the live result contract independent from persistence projections
and prevents a durability failure from masquerading as a successful run.

Persistent entities expose their own primary key as `id`. References to other
entities are qualified, such as `RunRecord.session_id` and
`HistoryItem.run_id`.

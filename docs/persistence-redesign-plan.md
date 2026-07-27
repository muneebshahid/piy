# Persistence-First Runtime Redesign

**Implementation status:** complete.
All eight units are implemented. This document records the design decisions
and implementation sequence rather than branch-specific progress.

## Purpose

Tile currently treats the live `Run` handle as authoritative and persists run
summaries and conversation history through separate, best-effort stores. This
allows the in-process outcome, stored run status, and stored history to
diverge.

This redesign makes persistence the lifecycle authority while preserving
Tile's existing execution guarantees:

- a submitted prompt is owned by a background run;
- event subscribers do not own execution;
- every in-process run log closes exactly once;
- only typed, replayable conversation items enter session history;
- provider and tool execution do not hold open database transactions.

The work is divided into eight independently verifiable units.

## Agreed decisions

### One store owns the consistency boundary

Replace `HistoryStore` and `RunStore` with one `Store` protocol. The protocol
exposes atomic lifecycle operations rather than independent table CRUD:

```python
class Store(Protocol):
    def create_session(*, record: SessionRecord) -> SessionRecord: ...
    def get_session(...) -> SessionRecord: ...
    def list_sessions(...) -> Sequence[SessionRecord]: ...

    def start_run(*, record: RunRecord, replace_active: bool) -> StartedRun: ...
    def finish_run(
        *,
        record: RunRecord,
        history_delta: Sequence[ConversationItem],
    ) -> RunRecord: ...

    def get_history(...) -> Sequence[HistoryItem]: ...
    def get_run(...) -> RunRecord: ...
    def list_runs(...) -> Sequence[RunRecord]: ...

    def fork_session(
        *,
        source_session_id: str,
        target: SessionRecord,
    ) -> SessionRecord: ...
```

The protocol must not expose unrestricted `update_run()` or
`append_history()` operations. Those methods allow callers to bypass run
ownership and cross-table invariants.

Atomicity is a semantic requirement, not a requirement to use SQL. SQLite can
use transactions. An append-only store could encode each lifecycle operation
as one journal record with locking, revisions, recovery, and idempotency. A
backend that cannot provide equivalent semantics is not a valid persistent
`Store`.

### Two short transactions surround execution

No database transaction remains open while a provider streams or a tool runs.

```text
start_run transaction
        ↓
provider and tool execution in memory
        ↓
finish_run transaction
```

`start_run` atomically validates session availability and records a running
run. `finish_run` atomically commits the terminal run outcome and its
replayable history.

### Submitted prompts belong to running records first

The submitted prompt is stored on `RunRecord` during `start_run`. It does not
enter committed session history until `finish_run`.

This keeps replaced or crashed attempts available for diagnosis without
allowing provisional user messages to pollute canonical replay history.

### Domain code uses typed conversation items

SQLite stores a history payload as JSON, but serialized dictionaries do not
escape the infrastructure adapter. Runtime and domain code use the existing
typed conversation union:

```python
ConversationItem = Annotated[
    UserMessage | AssistantTurn | ToolResultTurn,
    Field(discriminator="role"),
]
```

The history envelope is typed:

```python
class HistoryItem(BaseModel):
    """One immutable item in a session's committed conversation timeline."""

    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    run_id: str
    position: int
    item: ConversationItem
    created_at: datetime
```

`run_id` records the run that originally produced the item. Forked history
retains that provenance even though the copied item belongs to a new session.

### History remains flat

The MVP does not introduce history association tables, parent pointers, or
active tree heads. Each session owns a flat ordered set of history rows.

Forking duplicates history rows with new item IDs and the new session ID while
preserving:

- position;
- typed payload;
- original run ID;
- original creation timestamp.

Runs are not duplicated or reassigned. Immediately after a fork, the new
session has inherited history but no originating runs of its own.

### Status follows the terminal outcome

Terminal status is derived directly:

| Outcome | Status |
| --- | --- |
| `Completed` | `completed` |
| `Failed` | `failed` |
| `Aborted` | `aborted` |

`AgentFailure` remains distinct from runtime or persistence failure as a
structured cause, but both produce `status="failed"`. A model-declared failure
does not need to become an in-process exception.

### Replacement is not checkpoint resumption

The prompt API uses `replace_active`, not `resume`. Replacement starts a new
run and invalidates an old running attempt. Checkpoint resumption remains
future work.

`replace_active=True` means:

- replace the existing run if it is still running;
- otherwise start normally;
- reject every later database mutation from the replaced run.

## Target SQLite schema

The exact column serialization can evolve during implementation, but the
logical schema is:

```text
sessions
--------
id
name
created_at
updated_at

runs
----
id
session_id
prompt
status
started_at
ended_at
model
provider
outcome_json

history_items
-------------
id
session_id
run_id
position
role
payload_json
created_at

tile_meta
---------
key
value
```

Required constraints:

- `runs.session_id` references `sessions.id`;
- `history_items.session_id` references `sessions.id`;
- `history_items.run_id` references `runs.id`;
- `(session_id, position)` is unique;
- at most one run has `status="running"` for a session;
- terminal run fields must agree with the serialized outcome.

The one-running-run invariant should be enforced by a partial unique index:

```sql
CREATE UNIQUE INDEX one_running_run_per_session
ON runs(session_id)
WHERE status = 'running';
```

There is no migration from previous development schemas. Unsupported schema
versions fail clearly instead of being silently reinterpreted.

## Unit 1: Persistent domain records

### Changes

- Introduce or refine immutable `SessionRecord`, `RunRecord`, and
  `HistoryItem` models.
- Construct new records through `SessionRecord.create()` and
  `RunRecord.start()` before they cross the Store boundary.
- Add the submitted prompt to `RunRecord`.
- Rename the execution-facing `Run` to `RunHandle`.
- Make `ConversationItem` an explicit discriminated union.
- Derive terminal status directly from the outcome.
- Preserve distinct agent, runtime, and persistence failure causes.
- Extend aborted outcomes enough to distinguish explicit cancellation from
  replacement.
- Remove fields and convenience properties that are not required by the MVP
  after checking their public usage.

### Verification

- Model lifecycle validation.
- Frozen-record behavior.
- Typed conversation-item serialization round trips.
- Status/outcome consistency.
- Failure-cause serialization.

## Unit 2: Unified store and SQLite implementation

### Changes

- Introduce the use-case-oriented `Store` protocol.
- Accept caller-constructed records for session creation, run start and
  finish, and the target side of a fork.
- Implement one `SQLiteStore`, including `in_memory=True`.
- Create the unified schema and foreign keys.
- Add history ordering and active-run constraints.
- Keep raw SQL rows and JSON serialization inside the SQLite adapter.
- Remove the need for callers to combine arbitrary history and run stores.

### Verification

- Session, run, and typed history round trips.
- Foreign-key enforcement.
- Duplicate and missing-record failures.
- Only one running run per session.
- Unsupported schema versions fail clearly.
- File-backed state survives store restart.

## Unit 3: Atomic run submission

### Changes

Implement `Store.start_run()` as one transaction:

1. verify that the session exists;
2. check for a still-running run;
3. reject a busy session when replacement is disabled;
4. snapshot the committed session history used to bootstrap execution;
5. insert the supplied running record with its prompt and configured metadata;
6. commit before provider execution begins.

Remove `_active_prompt_session_ids` as the consistency authority. The runtime
may retain a local collection of handles only for cancellation and lifecycle
management.

### Verification

- Unknown sessions are rejected.
- Two runtimes sharing one store cannot start overlapping runs.
- A failed start creates no partial run.
- Provider execution cannot start before the running record commits.
- A process crash leaves a durable running record containing the prompt.

## Unit 4: Run-local execution and buffering

### Changes

- Make `RunHandle` own live events, its execution task, and a provisional
  conversation delta.
- Load committed session history once at run start.
- Build a run-local working history beginning with the submitted user prompt.
- Append completed assistant turns, tool results, and typed-result follow-ups
  to that working history.
- Stop projecting history events into the store during execution.
- Keep failed or aborted assistant turns out of the replayable delta.

Typed-result retries require particular care. They currently reload history
from the store and rely on intermediate persistence. After this change, every
attempt must receive the run-local working history so it can see previous
attempts without a database write.

### Verification

- Tool-loop requests see prior assistant and tool-result items.
- Typed-result retries see prior attempts and follow-ups.
- No history rows are written while the run is active.
- `Session.history` exposes committed history only.
- `RunReport.history_delta` exposes the terminal run-local history.
- Live events remain replayable to multiple subscribers.

## Unit 5: Atomic run finalization

### Changes

Before persistence, prepare a typed replayable history delta:

- prepend the submitted user prompt;
- retain completed assistant turns;
- include tool results;
- heal unanswered tool calls when appropriate.

`RunHandle` derives the terminal `RunRecord`, prepares this completion value,
and passes it to the
application-owned `on_finished` callback. `AgentRuntime` owns that callback:
it invokes `Store.finish_run()`, reconciles stale replacements, and returns
exactly one `RunRecord`. `RunHandle` never receives or calls a `Store`; it
builds one immutable `RunReport` from the returned record and closes its event
log. Store adapters translate backend-specific failures into
`StorePersistenceError`.

Implement `Store.finish_run()` as one transaction:

1. conditionally transition the run only when it remains `running`;
2. insert the complete flat history delta;
3. persist the terminal outcome and timestamp without rewriting the run's
   creation-time provider or model identity;
4. commit everything together.

The run-status condition is also the stale-writer fence:

```sql
WHERE id = ?
  AND status = 'running'
```

Zero affected rows raises `StaleRunError`, and no history is inserted.

History policy:

| Ending | Committed history |
| --- | --- |
| Completed | Full replayable delta |
| Agent-declared failure | Full replayable delta |
| Provider/runtime failure | Valid completed prefix |
| Explicit abort | Valid healed prefix |
| Replaced | Nothing |

If persistence fails, the handle preserves the candidate execution outcome in
a process-local terminal record and puts the Store error in
`RunReport.finalization_error`. `wait()` returns that report rather than
raising, and `RunReport.persisted` is false. The stored run remains `running`
because the transaction rolled back and requires explicit replacement.

### Verification

- Terminal outcome and history commit together.
- A history failure rolls back the run transition.
- A run-transition failure inserts no history.
- A replaced run cannot commit.
- Persistence failure is visible without replacing the execution outcome.
- Every in-process log still closes with exactly one `RunEndEvent`.

## Unit 6: Active-run replacement

### Changes

Add `replace_active` to prompt submission.

The start transaction:

1. finds the current running run;
2. rejects it when replacement is disabled;
3. when enabled, marks the still-running run as replaced;
4. inserts the replacement run;
5. commits both changes together.

If the previous run finished before this transaction acquired the write lock,
there is nothing to replace and the new run starts normally.

Return the new record, its transactionally consistent committed history
snapshot, and the optional replaced ID:

```python
class StartedRun(BaseModel):
    run: RunRecord
    committed_history: tuple[HistoryItem, ...]
    replaced_run_id: str | None
```

After commit, the runtime cancels the replaced local handle when it owns one.
Database fencing remains authoritative when the old handle lives elsewhere.

### Verification

- Old run finishes before replacement.
- Replacement commits before old finalization.
- Replacement when no run is active.
- Two simultaneous replacement attempts.
- Replacement across runtime instances.
- Replaced handles cannot append history or overwrite their stored outcome.

## Unit 7: Flat session forks

### Changes

Implement `Store.fork_session()` as one transaction:

1. validate the source and target IDs;
2. insert the supplied target session record;
3. read all committed source history;
4. insert duplicated history rows with new item IDs and the target session ID;
5. preserve positions, typed payloads, origin run IDs, and timestamps;
6. commit the session and copied history together.

The fork does not duplicate runs. Runs listed for a session are only runs that
originated in that session. Partial-history forks are deliberately deferred to
the future in-session history-tree work.

### Verification

- Fork gets a new session ID.
- Fork history equals the complete committed source history.
- History row IDs are distinct.
- Origin run provenance is preserved.
- Source and fork diverge independently.
- The new session has no own runs initially.
- A failed fork creates neither a session nor partial history.

## Unit 8: Public API, documentation, and cleanup

### Changes

Update construction:

```python
runtime = AgentRuntime(
    stream_fn=...,
    model=...,
    cwd=...,
    store=SQLiteStore(...),
)
```

Remove:

- `HistoryStore`;
- `RunStore`;
- separate in-memory and SQLite history/run stores;
- `_active_prompt_session_ids`;
- unrestricted history append and run update methods;
- the current best-effort `persistence_error` contract.

Expose:

- `Store`;
- `SQLiteStore`;
- `Session`;
- `RunHandle`;
- persistent record models;
- lifecycle and persistence errors.

Document:

- committed versus provisional history;
- atomic start and finish guarantees;
- status and outcome semantics;
- `replace_active`;
- flat fork copying;
- unsupported old schemas;
- custom-store atomicity requirements.

Update all examples, package exports, architecture documentation, and tests.

## Explicit non-goals

- No database migration.
- No in-session history tree or active history head.
- No checkpoint resumption.
- No worker lease or generation token.
- No persisted event replay.
- No automatic stale-run recovery.
- No conversation compaction changes.
- No history garbage collection.
- No run-to-many-session association.

Pi-style in-session history trees are tracked separately in
[TIL-60](https://linear.app/tileagent/issue/TIL-60/add-pi-style-in-session-history-trees).

## Definition of done

Every unit must preserve a working tree and finish with:

```bash
make format
make type_check
make test
```

Any violation of the persistence invariants must be represented by a typed
error and covered by an end-to-end store/runtime test.

# Issues

We are not persistence first at the moment. For Runs, we also have equivalent RunRecord, but no such thing for History.

We create a session table in history store but not in run. where both of them should be able to refer to a session as a foreign key. We need 3 tables,

1. Session table
2. Run table
3. History table

where both Run and History tables have a foreign key to the Session table.

Additionally we ask for HistoryStore and RunStore, which could lead to discrepancies one could be in memory and other in persistence. We should instead have a single store.

We should also have pydantic models or data classes that represent the data in the database (a bit like sqlalchemy models) such that operation on in memory objects also persists them. In case of failure in persistence, we should rollback the changes what we can, raise an exception and exit. Ideally once the issue is fixed, we can pick up from where we left off, but this could get complicated. Two issues

1. If a run was started and in running and then we fail to update it stays in running.
2. Currently we dont allow a running session to be run again while it is running. This means a session can get stuck in runninsg state.

We can introduce a resume flag, that overrides the check. When the flag is on, we mark the previous run as failed and start a new run. Later on we could introduce checkpoints to allow resuming a run from a checkpoint.

Now in case where the run is started and we start another run while the first one is still running, what should we do? We mark the previous run as failed. Meaning when the previous run's process tries to update the run or anything in the database we raise an error and exit, that the run is no longer valid. Thus we keep the second run as the valid one.

Now for the databasemodels:

```python
class Session(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime

class Run(BaseModel):
    id: str
    session_id: str  # Foreign key to Session
    status: str  # e.g., 'running', 'failed', 'completed'
    created_at: datetime
    updated_at: datetime
    model: str  # Model used for the run
    reasoning_effort: str  # Reasoning effort used for the run
    provider: str  # Provider used for the run


class HistoryItem(BaseModel):
    id: str
    session_id: str  # Foreign key to Session
    run_id: str  # Foreign key to Run
    data: dict | JSON  # Arbitrary data related to the history, serialized ConversationItem
    created_at: datetime
    updated_at: datetime
```

Then History could simply be a list of HistoryItem objects:

```python
class History(BaseModel):
    items: list[HistoryItem]
```

We can consider dropping the in memory store entirely and one can use sqlite in memory if they want to use an in memory store.

So then now the question is should we implement one giant store and expose it's Protocol for users to build their own?

`_active_prompt_session_ids` can be retired. As instead we now just look at status of the runs. And we only consider those history items valid where the corresponding run is not Failed for replay. Or another way could be we keep it all in memory and then persist at the end like now in one transaction!!!. But in case of failure raise loudly, log, and just accept the fact that we lost some data. What i don't want is the we keep the process running and then we have divergent state between in memory and persistence.

We could then introduce 3 service classes, one for each of the tables. Their whole job is persistence and retrieval of the data. We should also separate out current mixed in persistence logic in current Run class from pure execution, persistence can then be handled by the RunService. And we rename current Run class to RunHandle or something similar, which is pure execution.

Current AgentRuntime class can spit out as many session as it wants, but imo we restrict it as well one session per AgentRuntime.

There are 2 distinct designs we can consider for orchestration:

1. **Centralized Orchestration**: A single orchestrator service that manages the lifecycle of sessions, runs, and history. This service would handle creating sessions, starting runs, updating statuses, and persisting history items. It would ensure that all operations are atomic and consistent. This could be current Runtime class. Runtime class

2. **RunService as Orchestrator**: As it is really the RunService that drives the whole thing, one could hand over other 2 services to RunService and let it do the orchestration.

```python
class Runtime:
    def __init__(self,
                 cwd: str,
                 store: Store,
                 tools: list[Tool],
                 system_prompt: str,
                 autonomous_mode: bool):
        self.run_service = RunService(store)
        self.history_service = HistoryService(store)
        self.session_service = SessionService(store)
        self._session = self.session_service.create_session(name)
        self._run_handle = None

    def submit_prompt(self, stream_fn: Callable, prompt: str, resume: bool = False, result: type[BaseModel] | None = None,) -> RunHandle:
        # Logic to submit a prompt for the current session and create a new run
        run = self._start_run(prompt, resume=resume)


    def _start_run(self, prompt: str, resume: bool = False) -> Run:
        if resume:
            self.run_service.mark_previous_run_as_failed(self._session.id)
        try:
            run = self.run_service.create_run(self._session.id, model="default_model", provider="default_provider", reasoning_effort="default_effort")
            return RunHandle(run, prompt, self.history_service.get_history(self._session.id), stream_fn, on_done=self._end_run) # and other dependencies

        except Exception as e:
            if self.run_service.is_run_active(self._session.id):
                self.run_service.mark_previous_run_as_failed(self._session.id)
            # Handle exception, possibly logging and re-raising
            raise e

    def _end_run(self, run: Run):
        # Logic to end a run, marking it as completed or failed based on the outcome
        # Mark run as completed, persist all events only now in conversation, no persistence before this. and all of this is transactional, all or nothing. Later on when we have checkpoints we can think of something more sophisticated. In case we fail to persist, raise loudly. At worst we lose one turn and that is fine for now. Before presistence heal tool calls if needed. If the run was already marked as failed, just log a warning and exit. Don't overwrite.
        pass

    @transactional
    def fork(self, new_session_id: str = None, new_session_name: str = None) -> Session:
        # Logic to fork a session, creating a new session with the same history
        self.session_service.fork_session(self._session.id, new_session_id, new_session_name)
        self.history_service.fork_history(self._session.id, new_session_id)
        self.run_service.fork_runs(self._session.id, new_session_id)
        return self.session_service.get_session(new_session_id)

    @staticmethod
    @transactional
    def fork_from(store: Store, session_id: str, new_session_id: str = None, new_session_name: str = None) -> Session:
        # Logic to fork a session from a given store, creating a new session with the same history
        pass


class RunService:
    def create_run(self, session_id: str, *, model: str, provider: str, reasoning_effort:str) -> Run:
        # Logic to start a new run for a given session
        if self.is_run_active(session_id):
            raise RunActiveException("A run is already active for this session.")
        # Create and presist in db and return
        return Run(session_id=session_id)

    def is_run_active(self, session_id: str) -> bool:
        # Logic to check if there's an active run for the given session
        pass

    def mark_previous_run_as_failed(self, session_id: str):
        # Logic to mark the previous run as failed for the given session
        pass

    def get_run(self, run_id: str) -> Run:
        # Logic to retrieve a run by ID
        pass

class HistoryService:
    def add_history_item(self, session_id: str, run_id: str, data: dict) -> HistoryItem:
        # Logic to add a new history item for a given session and run
        pass

class SessionService:
    def create_session(self, name: str) -> Session:
        # Logic to create a new session and persist it
        pass

    def get_session(self, session_id: str) -> Session:
        # Logic to retrieve a session by ID
        pass
```

What run to presist? what is completed run? what is failed? What is aborted? What history to presist. This needs clarification, until this point in this document it's all mixed up and conflicting.

So currently i think if the history is not corrupted and can be replayed, then we should presist it. Simplest way i can think of is
we mark the run as completed/failed/aborted based on the outcome of the run and persist history after to healing the history. If for some reason we fail to persis the history or the run, we raise an error and at max lose one turn.

As long as run is in status running we don't have an outcome. Currently it is possible to have AgentFailure but still have status as completed, which in current semantics as it should be but can be confusing. I am thinking of simplifying it as well. Status is completed only and only if outcome is Completed be it _execute_plain or _execute_typed. In case of AgentFailure we raise an error in _execute_typed, catch it in the task finalize and treat it like any other errors. This would also collapse ExecutionFailure and AgentFailure into just Failed.

Additionally for an outsider these 3 string types can be confusing

ExecutionFailureOrigin: TypeAlias = Literal["submission", "turn", "execution"]

We could have Exception types for these and agent failures. and attach the exception_type in ExecutionFailure plus the actual exception in RunHandle. We keep them separate as we want to keep outcome serializable. and keep the Run object 1-1 to the database row. Although i wonder if there is a better way to do this?

One other reason for this complexity is that we have stuff or propeties in current code that aren't really needed atm. So in our reimplementation we want to keep it simple.

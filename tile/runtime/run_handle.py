"""Caller-facing control and observation handle for one live run."""

from collections.abc import AsyncIterator

from tile.events import AgentEvent
from tile.result import RunResult
from tile.runtime.run_execution import RunExecution


class RunHandle:
    """Expose one run without owning execution or persistence behavior."""

    def __init__(self, execution: RunExecution) -> None:
        """Bind the handle to its internal run execution."""

        self._execution = execution

    @property
    def id(self) -> str:
        """Return the stable persistent run id."""

        return self._execution.id

    @property
    def session_id(self) -> str:
        """Return the session that owns this run."""

        return self._execution.session_id

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield live run events through its terminal event."""

        async for event in self._execution.events():
            yield event

    async def wait(self) -> RunResult:
        """Wait for and return the run's outcome or harness fault."""

        return await self._execution.wait()

    def abort(self) -> None:
        """Request explicit cancellation of this run."""

        self._execution.abort()

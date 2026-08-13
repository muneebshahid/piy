"""Agent run event contracts.

Run lifecycle contract: every run log begins with ``RunStartEvent`` and
ends with one terminal event. ``RunEndEvent`` commits a durable outcome;
``RunFaultEvent`` exposes a failure to establish that durable ending.
Only the run start and one of those endings are guaranteed. Inner events nest as
``run ⊃ agent attempt ⊃ turn ⊃ message / tool executions``; failures and
cancellation can leave inner starts without their normal ends.
The terminal run event ends anything still open. A run end's outcome explains
why execution stopped; a run fault describes the durability failure. Provider
stream fragments carried by
``MessageUpdateEvent`` are message content rather than lifecycle scopes.
Hard process death is outside this contract because no process can emit
after it has stopped.
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from tile.result import RunOutcome
from tile.types.conversation import (
    AssistantTurn,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import StreamUpdateEvent
from tile.types.tool_execution import ToolExecutionOutcome
from tile.types.tools import (
    JsonObject,
)


class AgentEvent(BaseModel):
    """Base event emitted by the stateless agent runner."""

    type: str


type EmitFn = Callable[[AgentEvent], None]
"""Synchronous all-or-raise sink for one ordered agent event."""


class RunStartEvent(AgentEvent):
    """Marks the start of one prompt run.

    Emitted by the run itself before execution starts, so every
    run log begins with a run start on every path.
    """

    type: Literal["run_start"] = "run_start"


class RunEndEvent(AgentEvent):
    """Marks the end of one prompt run and commits its terminal outcome.

    This closes every durably finalized run, terminating any inner lifecycle
    that remains open. The outcome is the same discriminated value recorded
    on the durable run summary.
    """

    type: Literal["run_end"] = "run_end"
    outcome: RunOutcome


class RunFaultEvent(AgentEvent):
    """Marks a run whose harness could not durably finish its lifecycle."""

    type: Literal["run_fault"] = "run_fault"
    exception_type: str
    message: str


class AgentStartEvent(AgentEvent):
    """Marks the start of one stateless agent attempt.

    Attempts within one run are strictly sequential, so events carry no
    attempt label: position in the log identifies the attempt, with
    ``ResultFollowUpEvent`` separating result-contract attempts.
    """

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(AgentEvent):
    """Marks successful completion of one stateless agent attempt."""

    type: Literal["agent_end"] = "agent_end"


class ResultFollowUpEvent(AgentEvent):
    """Marks an injected reminder that the run must end with a result call."""

    type: Literal["result_follow_up"] = "result_follow_up"
    message: UserMessage


class TurnStartEvent(AgentEvent):
    """Marks the start of a single assistant turn."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent):
    """Marks the end of a single assistant turn."""

    type: Literal["turn_end"] = "turn_end"
    assistant_turn: AssistantTurn
    tool_executions: tuple[ToolExecutionOutcome, ...]

    @property
    def tool_result_turns(self) -> tuple[ToolResultTurn, ...]:
        """Return replayable tool result projections for this turn."""

        return tuple(execution.tool_result_turn for execution in self.tool_executions)


class MessageStartEvent(AgentEvent):
    """Marks the start of a message lifecycle event."""

    type: Literal["message_start"] = "message_start"
    response_id: str | None = None


class MessageUpdateEvent(AgentEvent):
    """Carries assistant streaming updates during a message."""

    type: Literal["message_update"] = "message_update"
    stream_event: StreamUpdateEvent


class MessageEndEvent(AgentEvent):
    """Marks the end of a message lifecycle event."""

    type: Literal["message_end"] = "message_end"
    assistant_turn: AssistantTurn


class ToolExecutionStartEvent(AgentEvent):
    """Marks the start of a tool execution."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutionEndEvent(AgentEvent):
    """Marks the end of a tool execution."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    outcome: ToolExecutionOutcome


type AgentRunEvent = (
    RunStartEvent
    | RunEndEvent
    | RunFaultEvent
    | AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
    | ResultFollowUpEvent
)

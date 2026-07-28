"""Agent run events and provider stream callable contracts.

Run lifecycle contract: every run log begins with ``RunStartEvent`` and
ends with exactly one ``RunEndEvent`` committing the terminal outcome,
for every in-process termination path. Only those two events are
guaranteed. The inner events are producer-emitted and nest as
``run ⊃ agent attempt ⊃ turn ⊃ message / tool executions``; a failure or
abort can tear a run down with inner scopes still open, so an inner start
may lack its end. Consumers apply one sweep rule: an end event ends
anything still open inside its scope, and ``RunEndEvent`` ends everything
— its outcome names why, exactly once. Provider stream fragments (text,
reasoning, and tool-call start/delta/end updates carried by
``MessageUpdateEvent``) are message content, not lifecycle scopes: the
containing message's end, or the sweep, terminates their outstanding
state. Hard process death is outside this contract — no process can
emit an event after it has stopped.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel

from tile.result import RunOutcome
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import StreamUpdateEvent
from tile.types.tool_execution import ToolExecutionOutcome
from tile.types.tools import (
    JsonObject,
    ToolDefinition,
)


class StreamFn(Protocol):
    """Callable that starts a provider stream from model-visible history.

    ``provider`` names the provider identity this callable streams through,
    declared once where the callable is constructed so run records can carry
    provider identity before the first message finalizes.
    """

    provider: str

    def __call__(
        self,
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        tools: Sequence[ToolDefinition] | None,
    ) -> Awaitable[AsyncEventStream]: ...


class AgentEvent(BaseModel):
    """Base event emitted by the stateless agent runner."""

    type: str


EmitFn: TypeAlias = Callable[[AgentEvent], None]
"""Synchronous sink for one ordered agent event."""


class RunStartEvent(AgentEvent):
    """Marks the start of one prompt run.

    Emitted by the run itself before execution starts, so every
    run log begins with a run start on every path.
    """

    type: Literal["run_start"] = "run_start"


class RunEndEvent(AgentEvent):
    """Marks the end of one prompt run and commits its terminal outcome.

    Exactly one run end closes every run, as its final event, ending
    every scope still open in the log. The outcome is the same
    discriminated value recorded on the durable run summary; its variant
    implies how execution terminated and names the cause of any scopes
    the run tore down open.
    """

    type: Literal["run_end"] = "run_end"
    outcome: RunOutcome


class AgentStartEvent(AgentEvent):
    """Marks the start of one stateless agent attempt.

    Attempts within one run are strictly sequential, so events carry no
    attempt label: position in the log identifies the attempt, with
    ``ResultFollowUpEvent`` separating typed-result retries.
    """

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(AgentEvent):
    """Marks the end of one stateless agent attempt.

    An attempt whose turn errored in-band still ends normally; the
    run-level verdict lives on ``RunEndEvent``.
    """

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


AgentRunEvent: TypeAlias = (
    RunStartEvent
    | RunEndEvent
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

"""Output contract protocol: tool names, prompt text, and run outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from tile.types.tools import JsonObject

COMPLETE_TOOL_NAME: Final = "complete"
FAIL_TOOL_NAME: Final = "fail"
MAX_RESULT_FOLLOW_UPS: Final = 8

RESULT_CONTRACT: Final = f"""\
This run must end with a result tool call.
- When the task is complete, call `{COMPLETE_TOOL_NAME}` with your final result as \
its arguments. The arguments are validated against the required schema; validation \
errors come back as tool errors you can correct.
- If you cannot complete the task or cannot produce a conforming result, call \
`{FAIL_TOOL_NAME}` with a clear `reason`.
- Plain text does not end the run. Only a `{COMPLETE_TOOL_NAME}` or \
`{FAIL_TOOL_NAME}` call does."""

RESULT_FOLLOW_UP: Final = (
    f"You ended your turn without calling `{COMPLETE_TOOL_NAME}` or "
    f"`{FAIL_TOOL_NAME}`. Call `{COMPLETE_TOOL_NAME}` with your final result, "
    f"or `{FAIL_TOOL_NAME}` with a reason if you cannot."
)

NO_RESULT_REASON: Final = (
    f"The model ended the run without calling `{COMPLETE_TOOL_NAME}` or "
    f"`{FAIL_TOOL_NAME}`."
)


class Completed(BaseModel):
    """Terminal outcome for a run that delivered its result.

    ``value`` carries assistant text for plain prompts and the validated result
    instance for result prompts. A serialized result outcome revalidates into
    its plain ``JsonObject`` form instead of the original model type.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["completed"] = "completed"
    value: str | JsonObject | SerializeAsAny[BaseModel] = Field(
        union_mode="left_to_right"
    )


type ExecutionFailureOrigin = Literal["turn", "execution"]
type AbortReason = Literal["cancelled"]


class AgentFailure(BaseModel):
    """The agent could not deliver the requested result.

    This includes an explicit ``fail`` result and exhaustion of the result
    contract after the allowed correction attempts.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["agent_failure"] = "agent_failure"
    reason: str


class ExecutionFailure(BaseModel):
    """Serializable diagnostics for a run whose execution failed.

    ``origin`` names the runtime boundary that failed. The exception is
    normalized into stable, serializable diagnostics for durable storage.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["execution_failure"] = "execution_failure"
    origin: ExecutionFailureOrigin
    exception_type: str
    message: str


type FailureCause = AgentFailure | ExecutionFailure


class Failed(BaseModel):
    """Terminal outcome for a run that could not deliver its result.

    The structured ``cause`` distinguishes an agent-declared failure from
    an execution failure instead of overloading optional fields.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["failed"] = "failed"
    cause: FailureCause = Field(discriminator="type")


class Aborted(BaseModel):
    """Terminal outcome for a cancelled run."""

    model_config = ConfigDict(frozen=True)

    type: Literal["aborted"] = "aborted"
    reason: AbortReason = "cancelled"


type RunOutcome = Completed | Failed | Aborted


@dataclass(frozen=True)
class Faulted:
    """Process-local result when the harness cannot ensure durability."""

    error: BaseException
    type: Literal["faulted"] = "faulted"


type RunResult = RunOutcome | Faulted

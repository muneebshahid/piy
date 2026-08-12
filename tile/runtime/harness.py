"""Single-session agent harness and run coordination."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from tile.extensions import Extension
from tile.extensions.registry import _register_extensions
from tile.providers.base import Provider
from tile.result import COMPLETE_TOOL_NAME, FAIL_TOOL_NAME
from tile.runtime.run_execution import RunExecution, _RunDependencies
from tile.runtime.run_handle import RunHandle
from tile.sessions import Session
from tile.tool_executor import ToolExecutor
from tile.tools.support.paths import normalize_cwd
from tile.types.tools import ToolDefinition, ToolFunction


class AgentHarness:
    """Configure and coordinate agent runs for a session."""

    def __init__(
        self,
        *,
        session: Session,
        cwd: Path | str,
        instructions: str,
        tools: Sequence[ToolDefinition] = (),
        extensions: Sequence[Extension] = (),
    ) -> None:
        """Configure one session with tools, instructions, and extensions."""

        registry = _register_extensions(extensions)
        _reject_reserved_tool_names(tools)
        normalized_cwd = normalize_cwd(cwd)
        self._session: Final = session
        self._run_hooks: Final = registry.build_run_hooks()
        self._run_observers: Final = registry.build_run_observers()
        self._run_dependencies: Final = _RunDependencies(
            instructions=instructions,
            cwd=normalized_cwd,
            tool_executor=ToolExecutor(_bind_cwd_tools(tools, normalized_cwd)),
        )

    @property
    def session(self) -> Session:
        """Return the single session bound to this harness."""

        return self._session

    async def prompt(
        self,
        prompt: str,
        *,
        provider: Provider,
        result_type: type[BaseModel] | None = None,
    ) -> RunHandle:
        """Durably accept a prompt and return its live run handle."""

        execution = await RunExecution.start(
            session=self._session,
            prompt=prompt,
            result_type=result_type,
            provider=provider,
            dependencies=self._run_dependencies,
            hooks=self._run_hooks,
            observers=self._run_observers,
        )
        return RunHandle(execution)


RESERVED_TOOL_NAMES: Final = (COMPLETE_TOOL_NAME, FAIL_TOOL_NAME)


def _reject_reserved_tool_names(tools: Sequence[ToolDefinition]) -> None:
    """Reject caller tools whose names the output contract reserves."""

    for tool in tools:
        if tool.name.lower() in RESERVED_TOOL_NAMES:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved by the harness for "
                "output contracts; rename the tool."
            )


def _bind_cwd_tools(
    tools: Sequence[ToolDefinition],
    cwd: Path,
) -> tuple[ToolDefinition, ...]:
    """Bind the harness cwd into every tool declaring a cwd parameter."""

    return tuple(
        _bind_cwd(tool, cwd) if _expects_cwd(tool.fn) else tool for tool in tools
    )


def _bind_cwd(tool: ToolDefinition, cwd: Path) -> ToolDefinition:
    """Return a tool copy whose function receives the harness cwd."""

    _reject_cwd_schema_property(tool)
    fn = partial(tool.fn, cwd=cwd)
    return tool.model_copy(update={"fn": fn})


def _expects_cwd(fn: ToolFunction) -> bool:
    """Return whether a tool function declares an explicit cwd parameter."""

    parameter = inspect.signature(fn).parameters.get("cwd")
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _reject_cwd_schema_property(tool: ToolDefinition) -> None:
    """Reject a schema that would expose the harness-injected cwd."""

    properties = tool.input_model.model_json_schema().get("properties", {})
    if "cwd" in properties:
        raise ValueError(
            f"Tool '{tool.name}' declares cwd in its input schema; cwd is "
            "harness-injected and must not be model-visible."
        )

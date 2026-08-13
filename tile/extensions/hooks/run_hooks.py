"""Run-scoped lifecycle hook orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from tile.extensions.hooks.before_run import (
    BeforeRunContext,
    BeforeRunHook,
    _BeforeRunExecution,
)


class RunHooks:
    """Invoke the lifecycle hooks registered for one run."""

    def __init__(self, *, before_run: Sequence[BeforeRunHook] = ()) -> None:
        """Freeze the supplied hooks in registration order."""

        self._before_run = tuple(_BeforeRunExecution(hook) for hook in before_run)

    async def before_run(self, context: BeforeRunContext) -> BeforeRunContext:
        """Apply every pre-admission hook in registration order."""

        current = context
        for execution in self._before_run:
            current = await execution.apply(current)
        return current

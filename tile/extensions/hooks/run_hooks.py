from __future__ import annotations

from collections.abc import Sequence

from tile.extensions.hooks.before_run import (
    BeforeRunContext,
    BeforeRunHook,
    _BeforeRunExecution,
)


class RunHooks:
    def __init__(self, *, before_run: Sequence[BeforeRunHook] = ()) -> None:
        self._before_run = tuple(_BeforeRunExecution(hook) for hook in before_run)

    async def before_run(self, context: BeforeRunContext) -> BeforeRunContext:
        current = context
        for execution in self._before_run:
            current = await execution.apply(current)
        return current

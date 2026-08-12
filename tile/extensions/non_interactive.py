"""Non-interactive execution instructions packaged as a Tile extension."""

from typing import Final

from tile.extensions.hooks import BeforeRunContext, BeforeRunResult
from tile.extensions.registry import ExtensionRegistry

_NON_INTERACTIVE_INSTRUCTIONS: Final = """\
You are operating inside Tile, a headless agent runtime. No one is watching the run \
and no one can answer questions mid-task.
- Work autonomously: never ask questions or wait for user input. When the task is \
ambiguous, pick the most reasonable interpretation and state the assumption in your \
final message.
- If you are blocked on something only the caller can provide, stop and name exactly \
what is missing.
- Your final message is the deliverable: everything the caller needs must be in it. \
Text emitted between tool calls may never be seen.
- Report outcomes faithfully: if a command or test fails, say so with the relevant \
output. Never present unverified work as done."""


class NonInteractive:
    """Instruct every run to proceed without caller interaction."""

    def register(self, registry: ExtensionRegistry) -> None:
        """Register the policy at the pre-admission instruction boundary."""

        registry.before_run(self.before_run)

    async def before_run(self, context: BeforeRunContext) -> BeforeRunResult:
        """Prepend non-interactive guidance to the complete system prompt."""

        return BeforeRunResult(
            system_prompt=(
                f"{_NON_INTERACTIVE_INSTRUCTIONS}\n\n{context.system_prompt}"
            ),
        )

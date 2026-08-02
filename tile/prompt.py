"""System prompt composition for the default coding agent."""

from datetime import date
from pathlib import Path

AUTO_MODE = """\
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

DEFAULT_INSTRUCTIONS = """\
You are an expert coding agent. You complete tasks using the tools available to you.
- Prefer dedicated tools over broader ones when applicable.
- Do what the task requires and nothing more; do not expand scope or fix unrelated \
issues.
- Never revert changes you did not make, and avoid destructive commands unless the \
task explicitly requires them.
- Reference code as file_path:line_number.
- Be concise."""

PROJECT_CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def read_project_context(cwd: Path) -> str:
    """Concatenate project context files found in the working directory."""

    parts = []
    for filename in PROJECT_CONTEXT_FILENAMES:
        path = cwd / filename
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)
    return "\n\n".join(parts)


def build_system_prompt(
    instructions: str,
    cwd: Path,
    *,
    auto_mode: bool = True,
) -> str:
    """Compose auto mode, instructions, project context, and environment lines."""

    environment = (
        # The prompt states the user's local calendar date, so a naive today()
        # is the correct source.
        f"Current date: {date.today().isoformat()}\nCurrent working directory: {cwd}"  # noqa: DTZ011
    )
    parts = [
        AUTO_MODE if auto_mode else None,
        instructions,
        read_project_context(cwd),
        environment,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())

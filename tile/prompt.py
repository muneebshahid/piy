"""System prompt composition for a caller-defined agent."""

from datetime import date
from pathlib import Path
from typing import Final

PROJECT_CONTEXT_FILENAMES: Final = ("AGENTS.md", "CLAUDE.md")


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
) -> str:
    """Compose core instructions, project context, and environment lines."""

    environment = (
        # The prompt states the user's local calendar date, so a naive today()
        # is the correct source.
        f"Current date: {date.today().isoformat()}\nCurrent working directory: {cwd}"  # noqa: DTZ011
    )
    parts = [
        instructions,
        read_project_context(cwd),
        environment,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())

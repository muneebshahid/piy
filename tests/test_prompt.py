"""Tests for system prompt composition and project context discovery."""

from datetime import date
from pathlib import Path

from tile.prompt import (
    AUTO_MODE,
    build_system_prompt,
    read_project_context,
)


def _environment(cwd: Path) -> str:
    """Return the environment block expected at the end of every prompt."""

    return f"Current date: {date.today().isoformat()}\nCurrent working directory: {cwd}"  # noqa: DTZ011


def test_build_system_prompt_composes_all_tiers(tmp_path: Path) -> None:
    """Order the prompt as auto mode, instructions, project context, environment."""

    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")

    prompt = build_system_prompt("Instructions body.", tmp_path)

    assert prompt == (
        f"{AUTO_MODE}\n\nInstructions body.\n\nProject rules.\n\n"
        f"{_environment(tmp_path)}"
    )


def test_build_system_prompt_omits_disabled_auto_mode(tmp_path: Path) -> None:
    """Drop the auto-mode tier entirely when the caller disables it."""

    prompt = build_system_prompt("Instructions body.", tmp_path, auto_mode=False)

    assert prompt == f"Instructions body.\n\n{_environment(tmp_path)}"


def test_read_project_context_concatenates_context_files(tmp_path: Path) -> None:
    """Join AGENTS.md and CLAUDE.md contents in a stable order."""

    (tmp_path / "CLAUDE.md").write_text("Claude notes.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Agent rules.\n", encoding="utf-8")

    assert read_project_context(tmp_path) == "Agent rules.\n\nClaude notes."


def test_read_project_context_skips_missing_and_blank_files(tmp_path: Path) -> None:
    """Return an empty string when no context file has content."""

    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")

    assert read_project_context(tmp_path) == ""

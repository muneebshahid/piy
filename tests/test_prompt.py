"""Tests for system prompt composition and project context discovery."""

from datetime import datetime
from pathlib import Path

from tile.prompt import build_system_prompt, read_project_context


def _environment(cwd: Path) -> str:
    return f"Current date: {datetime.now().astimezone().date().isoformat()}\nCurrent working directory: {cwd}"


def test_build_system_prompt_composes_all_core_tiers(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")

    prompt = build_system_prompt("Instructions body.", tmp_path)

    assert prompt == (
        f"Instructions body.\n\nProject rules.\n\n{_environment(tmp_path)}"
    )


def test_read_project_context_concatenates_context_files(tmp_path: Path) -> None:
    """Join AGENTS.md and CLAUDE.md contents in a stable order."""

    (tmp_path / "CLAUDE.md").write_text("Claude notes.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Agent rules.\n", encoding="utf-8")

    assert read_project_context(tmp_path) == "Agent rules.\n\nClaude notes."


def test_read_project_context_skips_missing_and_blank_files(tmp_path: Path) -> None:
    """Return an empty string when no context file has content."""

    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")

    assert read_project_context(tmp_path) == ""

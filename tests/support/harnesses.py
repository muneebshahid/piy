"""Shared harness builders for session-bound runtime tests."""

from collections.abc import Sequence
from pathlib import Path

from tile import AgentHarness, SessionRepository
from tile.extensions import Extension, NonInteractive
from tile.store import SQLiteStore
from tile.types.tools import ToolDefinition

DEFAULT_TEST_EXTENSIONS: tuple[Extension, ...] = (NonInteractive(),)


def build_harness(
    store: SQLiteStore,
    *,
    session_id: str | None = None,
    instructions: str = "Test agent.",
    tools: Sequence[ToolDefinition] = (),
    cwd: Path | str = Path(),
    extensions: Sequence[Extension] = DEFAULT_TEST_EXTENSIONS,
) -> AgentHarness:
    """Build a harness bound to one fresh session on the supplied store."""

    session = SessionRepository(store).create(session_id=session_id)
    return AgentHarness(
        session=session,
        cwd=cwd,
        instructions=instructions,
        tools=tools,
        extensions=extensions,
    )

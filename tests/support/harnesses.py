"""Shared harness builders for session-bound runtime tests."""

from collections.abc import Sequence
from pathlib import Path

from tile import AgentHarness, SessionRepository
from tile.store import SQLiteStore
from tile.types.tools import ToolDefinition


def build_harness(
    store: SQLiteStore,
    *,
    session_id: str | None = None,
    tools: Sequence[ToolDefinition] = (),
    cwd: Path | str = Path("."),
    auto_mode: bool = True,
) -> AgentHarness:
    """Build a harness bound to one fresh session on the supplied store."""

    session = SessionRepository(store).create(session_id=session_id)
    return AgentHarness(
        session=session,
        cwd=cwd,
        tools=tools,
        auto_mode=auto_mode,
    )

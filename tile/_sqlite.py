"""Shared SQLite connection scaffolding for Tile's durable Store."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_IN_MEMORY_DATABASE = ":memory:"


def resolve_connection_target(
    *,
    database_path: Path | str | None,
    in_memory: bool,
) -> str:
    """Return the SQLite connection target for file or in-memory mode."""

    if in_memory:
        return _IN_MEMORY_DATABASE
    if database_path is None:
        raise ValueError("database_path is required unless in_memory=True.")
    return str(Path(database_path).expanduser())


@contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run a block inside an immediate SQLite write transaction."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()

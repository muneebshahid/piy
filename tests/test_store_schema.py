"""Schema compatibility and constraint tests for the SQLite Store."""

import sqlite3
from pathlib import Path

import pytest

from tile.store import SQLiteStore, SQLiteStoreSchemaError
from tests.support.store import STARTED_AT, create_session


def test_unified_store_rejects_legacy_schema_without_migration(
    tmp_path: Path,
) -> None:
    """Fail clearly rather than reinterpret split-store development data."""

    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('schema_version', '2')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match="pre-unified"):
        SQLiteStore(database_path)


def test_unified_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Reject data written by an unsupported future unified schema."""

    database_path = tmp_path / "future.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES ('store_schema_version', '999')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match="999"):
        SQLiteStore(database_path)


def test_unified_schema_declares_required_foreign_keys(tmp_path: Path) -> None:
    """Tie runs and history to their persistent aggregate records."""

    database_path = tmp_path / "constraints.db"
    store = SQLiteStore(database_path)
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        run_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(runs)"
        ).fetchall()
        history_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(history_items)"
        ).fetchall()
    finally:
        connection.close()

    assert {(row[2], row[3], row[4]) for row in run_foreign_keys} == {
        ("sessions", "session_id", "id")
    }
    assert {(row[2], row[3], row[4]) for row in history_foreign_keys} == {
        ("runs", "run_id", "id"),
        ("sessions", "session_id", "id"),
    }


def test_unified_schema_uses_id_for_entity_primary_keys(tmp_path: Path) -> None:
    """Use id for each entity and qualified names for foreign keys."""

    database_path = tmp_path / "identifiers.db"
    store = SQLiteStore(database_path)
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        primary_keys = {
            table: _primary_key_column(connection, table)
            for table in ("sessions", "runs", "history_items")
        }
    finally:
        connection.close()

    assert primary_keys == {
        "sessions": "id",
        "runs": "id",
        "history_items": "id",
    }


def test_unified_schema_keeps_only_session_identity_and_timestamps(
    tmp_path: Path,
) -> None:
    """Exclude unused display metadata from persistent session records."""

    database_path = tmp_path / "session-columns.db"
    store = SQLiteStore(database_path)
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    finally:
        connection.close()

    assert columns == {"id", "created_at", "updated_at"}


def test_unified_schema_enforces_run_session_foreign_key(tmp_path: Path) -> None:
    """Reject a run whose owning session does not exist."""

    database_path = tmp_path / "run-foreign-key.db"
    store = SQLiteStore(database_path)
    store.close()
    connection = _foreign_key_connection(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, prompt, status, started_at,
                    ended_at, model, provider, outcome_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "orphan",
                    "missing",
                    "hello",
                    "running",
                    STARTED_AT.isoformat(),
                    None,
                    "gpt-5.4",
                    "test",
                    None,
                ),
            )
    finally:
        connection.close()


def test_unified_schema_enforces_history_run_foreign_key(tmp_path: Path) -> None:
    """Reject history whose originating run does not exist."""

    database_path = tmp_path / "history-foreign-key.db"
    store = SQLiteStore(database_path)
    create_session(store, session_id="session-1")
    store.close()
    connection = _foreign_key_connection(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO history_items (
                    id, session_id, run_id, position,
                    role, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "history-1",
                    "session-1",
                    "missing",
                    0,
                    "user",
                    '{"role":"user","content":"hello"}',
                    STARTED_AT.isoformat(),
                ),
            )
    finally:
        connection.close()


def test_unified_schema_requires_run_provider_identity(tmp_path: Path) -> None:
    """Reject a persistent run whose provider was not known at creation."""

    database_path = tmp_path / "provider-constraint.db"
    store = SQLiteStore(database_path)
    create_session(store, session_id="session-1")
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, prompt, status, started_at,
                    ended_at, model, provider, outcome_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "invalid",
                    "session-1",
                    "hello",
                    "running",
                    STARTED_AT.isoformat(),
                    None,
                    "gpt-5.4",
                    None,
                    None,
                ),
            )
    finally:
        connection.close()


def test_unified_schema_rejects_inconsistent_run_lifecycle_rows(
    tmp_path: Path,
) -> None:
    """Enforce basic terminal-field agreement below the domain adapter."""

    database_path = tmp_path / "lifecycle-constraint.db"
    store = SQLiteStore(database_path)
    create_session(store, session_id="session-1")
    store.close()
    connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, prompt, status, started_at,
                    ended_at, model, provider, outcome_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "invalid",
                    "session-1",
                    "hello",
                    "completed",
                    STARTED_AT.isoformat(),
                    None,
                    "gpt-5.4",
                    "test",
                    None,
                ),
            )
    finally:
        connection.close()


def _primary_key_column(connection: sqlite3.Connection, table: str) -> str:
    """Return the declared primary-key column for one table."""

    row = connection.execute(
        "SELECT name FROM pragma_table_info(?) WHERE pk = 1",
        (table,),
    ).fetchone()
    assert row is not None
    column = row[0]
    assert isinstance(column, str)
    return column


def _foreign_key_connection(database_path: Path) -> sqlite3.Connection:
    """Open a raw SQLite connection with foreign-key enforcement enabled."""

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    enabled = connection.execute("PRAGMA foreign_keys").fetchone()
    assert enabled == (1,)
    return connection

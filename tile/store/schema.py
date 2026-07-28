"""Creation and compatibility validation for the unified SQLite schema."""

import sqlite3
from typing import cast

from tile._sqlite import immediate_transaction
from tile.store.base import StoreError

SQLITE_STORE_SCHEMA_VERSION = "1"
_SCHEMA_VERSION_KEY = "store_schema_version"
_LEGACY_VERSION_KEYS = ("schema_version", "run_schema_version")


class SQLiteStoreSchemaError(StoreError):
    """Raised when a database does not use the supported unified schema."""


def initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    """Create the supported schema or reject an incompatible database."""

    with immediate_transaction(connection):
        _create_meta_table(connection)
        version = _read_schema_version(connection)
        if version is None:
            _reject_incompatible_unversioned_schema(connection)
            _create_schema(connection)
            _write_schema_version(connection)
            return
        if version != SQLITE_STORE_SCHEMA_VERSION:
            raise SQLiteStoreSchemaError(
                "Unsupported SQLite store schema version: "
                f"{version}. Expected {SQLITE_STORE_SCHEMA_VERSION}."
            )
        _create_schema(connection)


def _create_meta_table(connection: sqlite3.Connection) -> None:
    """Create the database metadata table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tile_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _read_schema_version(connection: sqlite3.Connection) -> str | None:
    """Return the unified schema version when present."""

    row = connection.execute(
        "SELECT value FROM tile_meta WHERE key = ?",
        (_SCHEMA_VERSION_KEY,),
    ).fetchone()
    return None if row is None else cast("tuple[str]", row)[0]


def _reject_incompatible_unversioned_schema(
    connection: sqlite3.Connection,
) -> None:
    """Reject removed split schemas and unrelated populated databases."""

    legacy_key = _legacy_version_key(connection)
    if legacy_key is not None:
        raise SQLiteStoreSchemaError(
            "Unsupported pre-unified Tile schema "
            f"({legacy_key}); create a new database."
        )
    table_name = _existing_application_table(connection)
    if table_name is not None:
        raise SQLiteStoreSchemaError(
            "Cannot initialize the unified Store over an unversioned "
            f"database containing table {table_name!r}."
        )


def _legacy_version_key(connection: sqlite3.Connection) -> str | None:
    """Return a removed split-store version key when present."""

    placeholders = ", ".join("?" for _ in _LEGACY_VERSION_KEYS)
    row = connection.execute(
        f"SELECT key FROM tile_meta WHERE key IN ({placeholders}) LIMIT 1",
        _LEGACY_VERSION_KEYS,
    ).fetchone()
    return None if row is None else cast("tuple[str]", row)[0]


def _existing_application_table(connection: sqlite3.Connection) -> str | None:
    """Return an existing non-metadata table when present."""

    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT IN ('tile_meta', 'sqlite_sequence')
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else cast("tuple[str]", row)[0]


def _write_schema_version(connection: sqlite3.Connection) -> None:
    """Record the current unified schema version."""

    connection.execute(
        "INSERT INTO tile_meta (key, value) VALUES (?, ?)",
        (_SCHEMA_VERSION_KEY, SQLITE_STORE_SCHEMA_VERSION),
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create session, run, history, and index schema objects."""

    _create_session_table(connection)
    _create_run_table(connection)
    _create_history_table(connection)
    _create_indexes(connection)


def _create_session_table(connection: sqlite3.Connection) -> None:
    """Create the session aggregate root table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _create_run_table(connection: sqlite3.Connection) -> None:
    """Create the persistent run table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            outcome_json TEXT,
            CHECK (
                (
                    status = 'running'
                    AND ended_at IS NULL
                    AND outcome_json IS NULL
                )
                OR (
                    status IN ('completed', 'failed', 'aborted')
                    AND ended_at IS NOT NULL
                    AND outcome_json IS NOT NULL
                )
            ),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """
    )


def _create_history_table(connection: sqlite3.Connection) -> None:
    """Create the flat committed history table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS history_items (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            role TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (role IN ('user', 'assistant', 'tool_result')),
            UNIQUE (session_id, position),
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
        """
    )


def _create_indexes(connection: sqlite3.Connection) -> None:
    """Create lookup and active-run invariant indexes."""

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS runs_session_id
        ON runs (session_id)
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_session
        ON runs (session_id)
        WHERE status = 'running'
        """
    )

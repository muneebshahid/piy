"""Schema compatibility and constraint tests for the SQLite Store."""

import sqlite3
from pathlib import Path

import pytest

from tests.support.store import STARTED_AT, create_session
from tile.store import SQLiteStore, SQLiteStoreSchemaError


def _raw_schema_connection(
    database_path: Path,
    *,
    session_id: str | None = None,
) -> sqlite3.Connection:
    """Initialize the unified schema, then reopen it as a raw connection."""

    store = SQLiteStore(database_path)
    try:
        if session_id is not None:
            create_session(store, session_id=session_id)
    finally:
        store.close()
    return sqlite3.connect(database_path)


def _insert_raw_run_row(
    connection: sqlite3.Connection,
    **overrides: str | None,
) -> None:
    """Insert one raw run row built from valid defaults plus overrides."""

    row: dict[str, str | None] = {
        "id": "invalid",
        "session_id": "session-1",
        "prompt": "hello",
        "status": "active",
        "started_at": STARTED_AT.isoformat(),
        "ended_at": None,
        "model": "gpt-5.4",
        "provider": "test",
        "outcome_json": None,
    }
    row.update(overrides)
    connection.execute(
        """
        INSERT INTO runs (
            id, session_id, prompt, status, started_at,
            ended_at, model, provider, outcome_json
        )
        VALUES (
            :id, :session_id, :prompt, :status, :started_at,
            :ended_at, :model, :provider, :outcome_json
        )
        """,
        row,
    )


@pytest.mark.parametrize(
    ("meta_key", "meta_value", "match"),
    [
        pytest.param("schema_version", "2", "pre-unified", id="legacy-split-schema"),
        pytest.param("store_schema_version", "999", "999", id="unknown-future-version"),
    ],
)
def test_unified_store_rejects_incompatible_schema_versions(
    tmp_path: Path,
    meta_key: str,
    meta_value: str,
    match: str,
) -> None:
    """Fail clearly rather than reinterpret data from unsupported schemas."""

    database_path = tmp_path / "incompatible.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE tile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO tile_meta (key, value) VALUES (?, ?)",
            (meta_key, meta_value),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match=match):
        SQLiteStore(database_path)


def test_unified_store_rejects_an_unversioned_database_with_foreign_tables(
    tmp_path: Path,
) -> None:
    """Refuse to build the schema inside an unrelated populated database."""

    database_path = tmp_path / "foreign.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE notes (id TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteStoreSchemaError, match="notes"):
        SQLiteStore(database_path)


def test_unified_schema_declares_required_foreign_keys(tmp_path: Path) -> None:
    """Tie runs and history to their persistent aggregate records."""

    connection = _raw_schema_connection(tmp_path / "constraints.db")
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


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"provider": None}, id="null-provider"),
        pytest.param({"status": "completed"}, id="terminal-status-without-end"),
    ],
)
def test_unified_schema_rejects_invalid_run_rows(
    tmp_path: Path,
    overrides: dict[str, str | None],
) -> None:
    """Enforce run identity and lifecycle agreement below the domain adapter."""

    connection = _raw_schema_connection(
        tmp_path / "run-constraints.db",
        session_id="session-1",
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _insert_raw_run_row(connection, **overrides)
    finally:
        connection.close()

"""Persistent domain records and the atomic Store contract."""

from tile.store.base import (
    ActiveRunError,
    InvalidHistoryError,
    RunAlreadyEndedError,
    RunAlreadyExistsError,
    RunNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    Store,
    StoreError,
    StoreOperation,
    StorePersistenceError,
)
from tile.store.models import (
    ActiveRun,
    HistoryItem,
    RunRecord,
    RunStatus,
    SessionRecord,
    StartedRun,
    TerminalRun,
    TerminalRunStatus,
)
from tile.store.schema import (
    SQLITE_STORE_SCHEMA_VERSION,
    SQLiteStoreSchemaError,
)
from tile.store.sqlite import SQLiteStore

__all__ = [
    "SQLITE_STORE_SCHEMA_VERSION",
    "ActiveRun",
    "ActiveRunError",
    "HistoryItem",
    "InvalidHistoryError",
    "RunAlreadyEndedError",
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "RunRecord",
    "RunStatus",
    "SQLiteStore",
    "SQLiteStoreSchemaError",
    "SessionAlreadyExistsError",
    "SessionNotFoundError",
    "SessionRecord",
    "StartedRun",
    "Store",
    "StoreError",
    "StoreOperation",
    "StorePersistenceError",
    "TerminalRun",
    "TerminalRunStatus",
]

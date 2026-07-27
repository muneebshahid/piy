"""Persistent domain records and the atomic Store contract."""

from tile.store.base import (
    ActiveRunError,
    InvalidHistoryError,
    RunAlreadyExistsError,
    RunNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    StaleRunError,
    Store,
    StoreError,
    StoreOperation,
    StorePersistenceError,
)
from tile.store.models import (
    HistoryItem,
    RunRecord,
    RunStatus,
    SessionRecord,
    StartedRun,
    TerminalRunStatus,
)
from tile.store.schema import (
    SQLITE_STORE_SCHEMA_VERSION,
    SQLiteStoreSchemaError,
)
from tile.store.sqlite import SQLiteStore

__all__ = [
    "ActiveRunError",
    "HistoryItem",
    "InvalidHistoryError",
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "RunRecord",
    "RunStatus",
    "SQLITE_STORE_SCHEMA_VERSION",
    "SQLiteStore",
    "SQLiteStoreSchemaError",
    "SessionAlreadyExistsError",
    "SessionNotFoundError",
    "SessionRecord",
    "StaleRunError",
    "StartedRun",
    "Store",
    "StoreError",
    "StoreOperation",
    "StorePersistenceError",
    "TerminalRunStatus",
]

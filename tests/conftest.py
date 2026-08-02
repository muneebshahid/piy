"""Shared fixtures for the tile test suite."""

from collections.abc import Iterator

import pytest

from tile.store import SQLiteStore


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    """Yield an in-memory SQLite Store closed on teardown."""

    store = SQLiteStore(in_memory=True)
    try:
        yield store
    finally:
        store.close()

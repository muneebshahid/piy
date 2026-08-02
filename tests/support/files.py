"""Newline-preserving file IO helpers for tool tests."""

from pathlib import Path


def write_text(path: Path, content: str) -> None:
    """Write exact file content without newline translation."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def read_text(path: Path) -> str:
    """Read exact file content without newline translation."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()

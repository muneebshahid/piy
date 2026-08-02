"""Tests for external executable availability helpers."""

import sys
from pathlib import Path

import pytest

from tests.support.command_mocks import (
    make_executable_available,
    make_executable_missing,
)
from tile.tools.support import executables
from tile.types.tools import ToolError


def test_require_executable_returns_resolved_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the executable path when the command is available."""

    make_executable_available(monkeypatch, "rg", "/usr/bin/rg")

    assert executables.require_executable("rg", "ripgrep (rg)") == "/usr/bin/rg"


def test_require_executable_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when the command is unavailable."""

    make_executable_missing(monkeypatch)

    with pytest.raises(ToolError, match="ripgrep"):
        executables.require_executable("rg", "ripgrep (rg)")


async def test_execute_runs_process_from_supplied_cwd(tmp_path: Path) -> None:
    """Run a process from the supplied working directory."""

    result = await executables.execute(
        sys.executable,
        ["-c", "from pathlib import Path; print(Path.cwd())"],
        cwd=tmp_path,
    )

    assert result == f"{tmp_path}\n"


async def test_execute_kills_process_at_stdout_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop collecting stdout at the cap and return complete lines."""

    monkeypatch.setattr(executables, "STDOUT_BYTE_CAP", 8192)

    result = await executables.execute(
        sys.executable,
        ["-c", "import sys\nwhile True: sys.stdout.write('x' * 1023 + '\\n')"],
        cwd=Path.cwd(),
    )

    assert result
    assert result.endswith("\n")
    assert all(line == "x" * 1023 for line in result.splitlines())


async def test_execute_replaces_invalid_utf8_output() -> None:
    """Decode non-UTF-8 output with replacement characters instead of failing."""

    result = await executables.execute(
        sys.executable,
        ["-c", r"import sys; sys.stdout.buffer.write(b'bad \xff byte\n')"],
        cwd=Path.cwd(),
    )

    assert result == "bad � byte\n"


async def test_execute_raises_when_capped_output_has_no_line_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail loudly instead of returning empty output for a giant single line."""

    monkeypatch.setattr(executables, "STDOUT_BYTE_CAP", 4096)

    with pytest.raises(ToolError, match="without a complete line"):
        await executables.execute(
            sys.executable,
            ["-c", "import sys\nwhile True: sys.stdout.write('x' * 1024)"],
            cwd=Path.cwd(),
        )


async def test_execute_drains_stderr_larger_than_pipe_buffers() -> None:
    """Avoid deadlock when a process floods stderr before exiting cleanly."""

    result = await executables.execute(
        sys.executable,
        ["-c", "import sys; sys.stderr.write('e' * 300000); print('done')"],
        cwd=Path.cwd(),
    )

    assert result == "done\n"


async def test_execute_raises_on_disallowed_exit_code() -> None:
    """Raise stderr output when a process exits with a disallowed code."""

    with pytest.raises(ToolError, match="boom"):
        await executables.execute(
            sys.executable,
            ["-c", "import sys; print('boom', file=sys.stderr); sys.exit(2)"],
            cwd=Path.cwd(),
        )


async def test_execute_allows_configured_exit_code() -> None:
    """Return stdout when a process exits with an explicitly allowed code."""

    result = await executables.execute(
        sys.executable,
        ["-c", "import sys; print('empty'); sys.exit(1)"],
        allowed_exit_codes=(0, 1),
        cwd=Path.cwd(),
    )

    assert result == "empty\n"

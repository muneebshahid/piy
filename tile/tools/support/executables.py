"""External executable availability helpers for built-in tools."""

import asyncio
import shutil
from asyncio import StreamReader
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from tile.types.tools import ToolError

# Maximum stdout bytes collected before the subprocess is killed.
STDOUT_BYTE_CAP: Final[int] = 16 * 1024 * 1024
# Maximum stderr bytes kept for failure messages; the rest is drained.
STDERR_BYTE_CAP: Final[int] = 64 * 1024


def require_executable(command: str, display_name: str) -> str:
    """Return an executable path or raise a clear tool availability error."""

    executable = shutil.which(command)
    if executable is None:
        raise ToolError(f"{display_name} is not available.")
    return executable


async def execute(
    executable: str,
    args: Sequence[str],
    allowed_exit_codes: tuple[int, ...] = (0,),
    *,
    cwd: Path,
) -> str:
    """Run an executable asynchronously and return stdout capped in size."""

    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except OSError as error:
        raise ToolError(str(error)) from error
    stderr_task = asyncio.create_task(_read_capped_stderr(process.stderr))
    stdout_bytes, capped = await _read_capped_stdout(process.stdout)
    if capped:
        _kill_process(process)
    exit_code = await process.wait()
    stderr_bytes = await stderr_task

    if capped:
        return _decode_complete_lines(stdout_bytes, executable)
    if exit_code not in allowed_exit_codes:
        error = (
            stderr_bytes.decode(errors="replace").strip()
            or f"{executable} exited with code {exit_code}"
        )
        raise ToolError(error)
    return stdout_bytes.decode(errors="replace")


async def _read_capped_stdout(stream: StreamReader | None) -> tuple[bytes, bool]:
    """Read stdout until end of stream or the collection cap is reached."""

    if stream is None:
        return b"", False

    collected = bytearray()
    while chunk := await stream.read(65536):
        collected.extend(chunk)
        if len(collected) >= STDOUT_BYTE_CAP:
            return bytes(collected), True
    return bytes(collected), False


async def _read_capped_stderr(stream: StreamReader | None) -> bytes:
    """Read stderr to end of stream, keeping only the leading capped bytes."""

    if stream is None:
        return b""

    kept = bytearray()
    while chunk := await stream.read(65536):
        if len(kept) < STDERR_BYTE_CAP:
            kept.extend(chunk[: STDERR_BYTE_CAP - len(kept)])
    return bytes(kept)


def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Force-kill a process that exceeded the output cap."""

    if process.returncode is None:
        process.kill()


def _decode_complete_lines(data: bytes, executable: str) -> str:
    """Decode capped output up to its last complete line."""

    end = data.rfind(b"\n")
    if end == -1:
        raise ToolError(
            f"{executable} output exceeded {STDOUT_BYTE_CAP} bytes "
            "without a complete line"
        )
    return data[: end + 1].decode(errors="replace")

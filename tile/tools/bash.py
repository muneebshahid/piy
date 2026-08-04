"""Shell command tool for the default agent."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from tile.tool_truncation import ToolOutputDetails, Truncation
from tile.tools.support.output_accumulator import OutputAccumulator, OutputSnapshot
from tile.tools.support.truncation import (
    OUTPUT_BYTE_LIMIT_LABEL,
    format_size,
)
from tile.types.tools import (
    ToolDefinition,
    ToolDetails,
    ToolError,
    ToolInput,
    ToolResult,
)

# Timeout applied when the model omits one, so hung commands cannot wedge a run.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 120


class BashDetails(ToolDetails):
    """Shell command metadata for UI and persistence."""

    type: Literal["bash"] = "bash"
    output: ToolOutputDetails


class BashInput(ToolInput):
    """Model-controlled shell command arguments."""

    command: str = Field(description="Bash command to execute.")
    timeout: float | None = Field(
        default=None,
        description="Timeout in seconds. Defaults to 120 seconds when omitted.",
    )


async def fn(params: BashInput, *, cwd: Path) -> ToolResult:
    """Execute a shell command from the agent working directory."""

    try:
        resolved_cwd = _resolve_cwd(cwd)
        result = await _execute(
            params.command,
            _effective_timeout(params.timeout),
            resolved_cwd,
        )
        return _build_result(result)
    except OSError as error:
        raise ToolError(str(error)) from error


def _resolve_cwd(cwd: Path) -> Path:
    """Resolve and validate the shell working directory."""

    resolved_cwd = cwd.expanduser().resolve(strict=True)
    if not resolved_cwd.is_dir():
        raise NotADirectoryError(resolved_cwd)
    return resolved_cwd


async def _execute(
    command: str,
    timeout: float,
    cwd: Path,
) -> OutputSnapshot:
    """Execute a shell command and return captured output for successful exits."""

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        start_new_session=_supports_process_groups(),
    )
    snapshot, timed_out = await _wait_for_process(process, timeout)
    _raise_for_execution_failure(
        snapshot, process.returncode, timed_out=timed_out, timeout=timeout
    )
    return snapshot


async def _wait_for_process(
    process: asyncio.subprocess.Process,
    timeout: float,
) -> tuple[OutputSnapshot, bool]:
    """Wait for a shell process while enforcing the timeout."""

    output = OutputAccumulator()
    output_task = asyncio.create_task(_read_output(process, output))
    wait_task = asyncio.create_task(process.wait())
    timed_out = False

    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
    except TimeoutError:
        timed_out = True
        await _stop_timed_out_process(process, wait_task)
    finally:
        await output_task

    return output.finish(), timed_out


async def _read_output(
    process: asyncio.subprocess.Process,
    output: OutputAccumulator,
) -> None:
    """Read merged process output until the pipe closes."""

    if process.stdout is None:
        return

    while chunk := await process.stdout.read(4096):
        output.accumulate(chunk)


async def _stop_timed_out_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
) -> None:
    """Terminate a timed-out process and escalate if it does not exit."""

    _terminate_process(process)
    done, _ = await asyncio.wait({wait_task}, timeout=1)
    if not done:
        _kill_process(process)
        await wait_task


def _build_result(snapshot: OutputSnapshot) -> ToolResult:
    """Build shell output for successful command execution."""

    if not snapshot.content and not snapshot.truncation.truncated:
        return ToolResult.text("(no output)")

    if not snapshot.truncation.truncated:
        return ToolResult.text(snapshot.content)

    text = _append_status(
        snapshot.content,
        _truncation_notice(snapshot.truncation),
    )
    return ToolResult.text(text, details=_build_details(snapshot.truncation))


def _raise_for_execution_failure(
    snapshot: OutputSnapshot,
    exit_code: int | None,
    *,
    timed_out: bool,
    timeout: float,
) -> None:
    """Raise when shell execution timed out or exited unsuccessfully."""

    if timed_out:
        raise ToolError(
            _append_status(
                _build_error_output_text(snapshot),
                _timeout_status(timeout),
            )
        )
    if exit_code not in (0, None):
        raise ToolError(
            _append_status(
                _build_error_output_text(snapshot),
                f"Command exited with code {exit_code}",
            )
        )


def _build_error_output_text(snapshot: OutputSnapshot) -> str:
    """Format captured shell output for error messages."""

    if not snapshot.content and not snapshot.truncation.truncated:
        return ""

    if not snapshot.truncation.truncated:
        return snapshot.content
    return _append_status(
        snapshot.content,
        _truncation_notice(snapshot.truncation),
    )


def _build_details(truncation: Truncation) -> BashDetails:
    """Build bash details from output truncation metadata."""

    return BashDetails(output=ToolOutputDetails.from_truncation(truncation))


def _truncation_notice(truncation: Truncation) -> str:
    """Return a compact bash output truncation notice."""

    if truncation.edge_line_exceeds_limit:
        return f"[Output omitted: last line exceeds {OUTPUT_BYTE_LIMIT_LABEL} limit]"

    start_line = truncation.total_lines - truncation.output_lines + 1
    end_line = truncation.total_lines
    if truncation.truncated_by == "lines":
        return f"[Showing lines {start_line}-{end_line} of {truncation.total_lines}]"
    return (
        f"[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
        f"({format_size(truncation.max_bytes)} limit)]"
    )


def _append_status(output: str, status: str) -> str:
    """Append a command status line after captured output."""

    if output:
        return f"{output}\n\n{status}"
    return status


def _timeout_status(timeout: float) -> str:
    """Return a timeout failure status message."""

    return f"Command timed out after {timeout:g} seconds"


def _effective_timeout(timeout: float | None) -> float:
    """Return a positive model-supplied timeout or the default."""

    if timeout is None or timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Send a graceful termination signal to a process or process group."""

    if process.returncode is not None:
        return
    if _supports_process_groups() and process.pid is not None:
        _signal_process_group(process.pid, signal.SIGTERM)
        return
    process.terminate()


def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Force-kill a process or process group."""

    if process.returncode is not None:
        return
    if _supports_process_groups() and process.pid is not None:
        _signal_process_group(process.pid, signal.SIGKILL)
        return
    process.kill()


def _signal_process_group(pid: int, signal_number: signal.Signals) -> None:
    """Send a signal to a POSIX process group if it still exists."""

    try:
        os.killpg(pid, signal_number)
    except ProcessLookupError:
        return


def _supports_process_groups() -> bool:
    """Return whether subprocesses can be isolated in POSIX process groups."""

    return sys.platform != "win32"


tool = ToolDefinition(
    name="bash",
    description="Execute a bash command.",
    input_model=BashInput,
    fn=fn,
)

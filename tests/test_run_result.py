"""Tests for process-local run results and fault events."""

from typing import get_args

from tile.events import RunFaultEvent
from tile.result import Faulted, RunOutcome, RunResult


def test_run_result_adds_faulted_without_expanding_persisted_outcomes() -> None:
    """Keep durability faults outside the RunRecord outcome contract."""

    error = OSError("disk full")
    result = Faulted(error=error)

    assert result.error is error
    assert result.type == "faulted"
    assert Faulted not in get_args(RunOutcome)
    assert Faulted in get_args(RunResult)


def test_run_fault_event_carries_serializable_error_details() -> None:
    """Expose a live terminal fault without serializing its exception object."""

    event = RunFaultEvent(exception_type="OSError", message="disk full")

    assert event.model_dump() == {
        "type": "run_fault",
        "exception_type": "OSError",
        "message": "disk full",
    }

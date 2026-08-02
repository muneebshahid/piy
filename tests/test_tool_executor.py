"""Tests for model-requested tool execution."""

import asyncio
from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema

from tile.tool_executor import ToolExecutor
from tile.types.tool_execution import (
    ToolInputValidationFailure,
    ToolInvocationFailure,
)
from tile.types.tools import (
    JsonObject,
    ToolDefinition,
    ToolDetails,
    ToolError,
    ToolInput,
    ToolResult,
)
from tests.support.tool_definitions import CityInput, NoInput, city_tool
from tests.support.tool_results import tool_text


async def _get_weather(params: CityInput) -> ToolResult:
    """Return deterministic weather for a city."""

    return ToolResult.text(f"{params.city}: sunny")


async def _raise_error(params: CityInput) -> ToolResult:
    """Raise a deterministic tool error."""

    _ = params
    raise RuntimeError("boom")


async def _noop(params: BaseModel) -> ToolResult:
    """Return a fixed result for tools that take no arguments."""

    _ = params
    return ToolResult.text("ok")


def _sample_tool() -> ToolDefinition:
    """Build a deterministic tool definition."""

    return city_tool(
        "get_weather",
        "Return deterministic weather.",
        _get_weather,
    )


def _failing_tool() -> ToolDefinition:
    """Build a deterministic failing tool definition."""

    return city_tool(
        "fail_weather",
        "Raise deterministic weather failure.",
        _raise_error,
    )


def test_tool_definition_rejects_input_model_without_json_schema() -> None:
    """Fail tool construction before a provider receives an invalid schema."""

    class InvalidInput(ToolInput):
        """Input containing a callable that JSON Schema cannot represent."""

        callback: Callable[[], str]

    with pytest.raises(PydanticInvalidForJsonSchema, match="CallableSchema"):
        ToolDefinition(
            name="invalid",
            description="Cannot expose this input to a provider.",
            input_model=InvalidInput,
            fn=_noop,
        )


def test_tool_executor_rejects_function_without_input_parameter() -> None:
    """Fail runtime setup when a function cannot receive its input model."""

    async def invalid() -> ToolResult:
        """Expose an invalid callback without an input-model parameter."""

        return ToolResult.text("invalid")

    tool = city_tool(
        "invalid",
        "Function does not accept the validated input model.",
        invalid,
    )

    with pytest.raises(ValueError, match="must accept one validated input model"):
        ToolExecutor([tool])


def test_tool_executor_rejects_duplicate_names() -> None:
    """Refuse to register two tools with the same name."""

    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolExecutor([_sample_tool(), _sample_tool()])


async def test_tool_executor_executes_registered_tool() -> None:
    """Execute a registered tool and return a normalized outcome."""

    executor = ToolExecutor([_sample_tool()])

    outcome = await executor.execute(
        call_id="call_weather",
        tool_name="get_weather",
        arguments={"city": "Munich"},
    )

    assert outcome.tool_result_turn.call_id == "call_weather"
    assert outcome.tool_result_turn.tool_name == "get_weather"
    assert outcome.tool_result_turn.is_error is False
    assert tool_text(outcome.tool_result_turn) == "Munich: sunny"


async def test_tool_executor_preserves_nested_input_models() -> None:
    """Pass nested validated models into tool code without flattening them."""

    class Item(BaseModel):
        """One nested item consumed by the deterministic tool."""

        name: str

    class ItemsInput(ToolInput):
        """Model-controlled collection of nested items."""

        items: list[Item]

    captured: list[Item] = []

    async def inspect_items(params: ItemsInput) -> ToolResult:
        """Capture the concrete nested model received by the tool."""

        captured.extend(params.items)
        return ToolResult.text(params.items[0].name)

    executor = ToolExecutor(
        [
            ToolDefinition(
                name="inspect_items",
                description="Inspect nested items.",
                input_model=ItemsInput,
                fn=inspect_items,
            )
        ]
    )

    outcome = await executor.execute(
        call_id="call_items",
        tool_name="inspect_items",
        arguments={"items": [{"name": "first"}]},
    )

    assert len(captured) == 1
    assert isinstance(captured[0], Item)
    assert tool_text(outcome.tool_result_turn) == "first"


async def test_tool_executor_normalizes_missing_tool() -> None:
    """Return an error outcome when a requested tool is not registered."""

    executor = ToolExecutor()

    outcome = await executor.execute(
        call_id="call_missing",
        tool_name="missing_tool",
        arguments={},
    )

    assert outcome.tool_result_turn.call_id == "call_missing"
    assert outcome.tool_result_turn.tool_name == "missing_tool"
    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.tool_result_turn) == "Tool 'missing_tool' not found"


async def test_tool_executor_normalizes_tool_exception() -> None:
    """Return an error outcome when tool execution raises."""

    executor = ToolExecutor([_failing_tool()])

    outcome = await executor.execute(
        call_id="call_fail",
        tool_name="fail_weather",
        arguments={"city": "Munich"},
    )

    assert outcome.tool_result_turn.call_id == "call_fail"
    assert outcome.tool_result_turn.tool_name == "fail_weather"
    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.tool_result_turn) == "boom"
    details = outcome.details
    assert isinstance(details, ToolInvocationFailure)
    assert details.tool_name == "fail_weather"
    assert details.exception_type == "RuntimeError"
    assert details.message == "boom"
    assert "exception" not in details.model_dump()


async def test_tool_executor_does_not_normalize_cancellation() -> None:
    """Let task cancellation propagate through the tool boundary."""

    async def cancel(params: CityInput) -> ToolResult:
        """Raise cancellation from a deterministic tool."""

        _ = params
        raise asyncio.CancelledError

    executor = ToolExecutor(
        [city_tool("cancel", "Cancel deterministic execution.", cancel)]
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(
            call_id="call_cancel",
            tool_name="cancel",
            arguments={"city": "Munich"},
        )


@pytest.mark.parametrize(
    ("arguments", "location", "code"),
    [
        ({}, ("city",), "missing"),
        ({"city": 5}, ("city",), "string_type"),
        (
            {"city": "Munich", "unexpected": True},
            ("unexpected",),
            "extra_forbidden",
        ),
    ],
)
async def test_tool_executor_rejects_invalid_arguments_before_invocation(
    arguments: JsonObject,
    location: tuple[str, ...],
    code: str,
) -> None:
    """Return structured correction details without invoking invalid input."""

    calls: list[str] = []

    async def capture(params: CityInput) -> ToolResult:
        """Record valid invocations for the boundary assertion."""

        calls.append(params.city)
        return ToolResult.text(params.city)

    executor = ToolExecutor(
        [city_tool("weather", "Return deterministic weather.", capture)]
    )

    outcome = await executor.execute(
        call_id="call_invalid",
        tool_name="weather",
        arguments=arguments,
    )

    assert calls == []
    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.tool_result_turn).startswith(
        "Invalid arguments for tool 'weather':"
    )
    details = outcome.details
    assert isinstance(details, ToolInputValidationFailure)
    assert [(issue.location, issue.code) for issue in details.issues] == [
        (location, code)
    ]
    assert "input" not in details.model_dump()


class _ExpectedFailureDetails(ToolDetails):
    """Domain metadata for an intentionally raised tool failure."""

    type: Literal["expected_failure"] = "expected_failure"
    reason: str


async def test_tool_executor_normalizes_intentional_tool_error() -> None:
    """Convert an intentional tool exception into model-visible error data."""

    async def unavailable(params: CityInput) -> ToolResult:
        """Raise a deterministic handled failure."""

        raise ToolError(
            f"Weather unavailable for {params.city}",
            details=_ExpectedFailureDetails(reason="maintenance"),
        )

    executor = ToolExecutor(
        [city_tool("weather", "Return deterministic weather.", unavailable)]
    )

    outcome = await executor.execute(
        call_id="call_unavailable",
        tool_name="weather",
        arguments={"city": "Munich"},
    )

    assert outcome.tool_result_turn.is_error is True
    assert tool_text(outcome.tool_result_turn) == "Weather unavailable for Munich"
    assert isinstance(outcome.details, _ExpectedFailureDetails)


class _DatabaseDetails(ToolDetails):
    """User-defined details for a custom database tool."""

    type: str = "database"
    rows_scanned: int


async def _query_database(params: NoInput) -> ToolResult:
    """Return a result carrying user-defined details."""

    _ = params
    return ToolResult.text("2 rows", details=_DatabaseDetails(rows_scanned=2))


async def test_tool_executor_preserves_user_defined_details() -> None:
    """Carry and serialize user-defined tool details end to end."""

    tool = ToolDefinition(
        name="query_database",
        description="Query a database.",
        input_model=NoInput,
        fn=_query_database,
    )
    executor = ToolExecutor([tool])

    outcome = await executor.execute(
        call_id="call_db",
        tool_name="query_database",
        arguments={},
    )

    details = outcome.details
    assert isinstance(details, _DatabaseDetails)
    assert details.rows_scanned == 2
    assert details.model_dump() == {
        "type": "database",
        "rows_scanned": 2,
    }


@pytest.mark.parametrize("name", ["", "   ", " read", "read ", "\tread\n"])
def test_tool_definition_rejects_empty_or_padded_names(name: str) -> None:
    """Fail tool registration for empty or whitespace-padded names."""

    with pytest.raises(ValidationError, match="non-empty without surrounding"):
        ToolDefinition(
            name=name,
            description="Read a file.",
            input_model=NoInput,
            fn=_noop,
        )


@pytest.mark.parametrize(
    ("registered_name", "requested_name"),
    [
        pytest.param("Read", "read", id="uppercase-registered"),
        pytest.param("read", "Read", id="uppercase-requested"),
        pytest.param("read", " Read ", id="whitespace-padded-request"),
    ],
)
async def test_tool_executor_finds_tool_by_normalized_name(
    registered_name: str,
    requested_name: str,
) -> None:
    """Find a registered tool by case-insensitive, whitespace-stripped lookup."""

    tool = ToolDefinition(
        name=registered_name,
        description="Read a file.",
        input_model=NoInput,
        fn=_noop,
    )
    executor = ToolExecutor([tool])

    outcome = await executor.execute(
        call_id="call_read",
        tool_name=requested_name,
        arguments={},
    )

    assert outcome.tool_result_turn.is_error is False
    assert tool_text(outcome.tool_result_turn) == "ok"

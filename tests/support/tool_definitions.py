"""Shared tool definition builders for tests."""

from pydantic import BaseModel, Field

from tile.types.tools import ToolDefinition, ToolFunction, ToolInput, ToolResult


class CityInput(ToolInput):
    """Strict city input shared by deterministic test tools."""

    city: str = Field(description="The city to look up.")


class NoInput(ToolInput):
    """Strict empty input shared by no-argument test tools."""


class WeatherReport(BaseModel):
    """Typed result schema shared by result-contract tests."""

    city: str
    temp_c: float


async def city_text_fn(params: CityInput) -> ToolResult:
    """Return deterministic city text for shared tool fixtures."""

    return ToolResult.text(f"city={params.city}")


def city_tool(
    name: str,
    description: str,
    fn: ToolFunction,
) -> ToolDefinition:
    """Build a deterministic tool definition with a single city parameter."""

    return ToolDefinition(
        name=name,
        description=description,
        input_model=CityInput,
        fn=fn,
    )

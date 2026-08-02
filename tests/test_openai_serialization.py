from pydantic import TypeAdapter

from tile.providers.openai.serialization import (
    serialize_history_items,
    serialize_tools,
)
from tile.types.conversation import AssistantTurn, ToolResultTurn, UserMessage
from tile.types.stream_events import (
    ProviderMetadata,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)
from tile.types.tools import ToolImageContent, ToolResult
from tests.support.tool_definitions import city_text_fn, city_tool
from openai.types.responses.response_input_param import ResponseInputParam


def test_serialize_history_items_flattens_sample_thread() -> None:
    """Replay a full thread with reasoning signatures and message metadata."""

    serialized = serialize_history_items(_sample_poem_thread())

    assert serialized == _expected_poem_history_items()
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_skips_reasoning_without_replay_metadata() -> None:
    history = [
        AssistantTurn(
            blocks=[
                ReasoningBlock(summary_text="Think first."),
                TextBlock(text="Answer next."),
            ]
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0_1",
            "content": [
                {
                    "type": "output_text",
                    "text": "Answer next.",
                    "annotations": [],
                }
            ],
        },
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_replays_tool_calls_and_tool_results() -> None:
    serialized = serialize_history_items(_tool_call_history())

    assert serialized == _expected_tool_call_response_input()
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_replays_tool_result_images() -> None:
    """Serialize image tool results as OpenAI Responses content parts."""

    history = [
        ToolResultTurn(
            call_id="call_123",
            tool_name="read",
            content=ToolResult.image(
                "Read image file [image/png]",
                ToolImageContent(data="ZmFrZQ==", mime_type="image/png"),
            ).content,
            is_error=False,
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [
                {"type": "input_text", "text": "Read image file [image/png]"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,ZmFrZQ==",
                    "detail": "auto",
                },
            ],
        }
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_tools_maps_tool_definitions_to_function_tools() -> None:
    tools = [
        city_tool(
            "get_weather",
            "Return the current weather for a city.",
            city_text_fn,
        )
    ]

    assert serialize_tools(tools) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": tools[0].input_schema,
            "strict": False,
            "defer_loading": False,
        }
    ]


def _sample_poem_thread() -> list[UserMessage | AssistantTurn]:
    """Build a sample conversation with replayable and skipped turns."""

    return [
        UserMessage(content="Write a haiku about rain."),
        AssistantTurn(
            response_id="resp_123",
            blocks=[
                ReasoningBlock(
                    summary_text="Draft a short seasonal poem.",
                    provider_metadata=ProviderMetadata.from_values(
                        reasoning_signature=_sample_reasoning_signature(),
                    ),
                ),
                TextBlock(
                    text=_sample_poem_text(),
                    provider_metadata=ProviderMetadata.from_values(
                        message_id="msg_123",
                        phase="final_answer",
                    ),
                ),
            ],
        ),
        AssistantTurn(
            status="aborted",
            blocks=[
                ReasoningBlock(summary_text="This partial turn should be skipped.")
            ],
        ),
        UserMessage(content="Revise the second line."),
    ]


def _expected_poem_history_items() -> ResponseInputParam:
    """Build the expected OpenAI input items for the sample poem conversation."""

    return [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Write a haiku about rain."}],
        },
        {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [
                {"type": "summary_text", "text": "Draft a short seasonal poem."}
            ],
            "encrypted_content": "enc_123",
            "status": "completed",
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_123",
            "phase": "final_answer",
            "content": [
                {"type": "output_text", "text": _sample_poem_text(), "annotations": []}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Revise the second line."}],
        },
    ]


def _tool_call_history() -> list[AssistantTurn | ToolResultTurn]:
    """Build history with an assistant tool call and matching result."""

    return [
        AssistantTurn(
            blocks=[
                TextBlock(
                    text="Checking the weather.",
                    provider_metadata=ProviderMetadata.from_values(message_id="msg_0"),
                ),
                ToolCallBlock(
                    call_id="call_123",
                    name="get_weather",
                    arguments={"city": "Berlin"},
                    provider_metadata=ProviderMetadata.from_values(
                        provider_item_id="fc_123"
                    ),
                ),
            ]
        ),
        ToolResultTurn(
            call_id="call_123",
            tool_name="get_weather",
            content=ToolResult.text("Temperature: 14 C").content,
            is_error=False,
        ),
    ]


def _expected_tool_call_response_input() -> ResponseInputParam:
    """Build expected OpenAI input for tool call history."""

    return [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0",
            "content": [
                {
                    "type": "output_text",
                    "text": "Checking the weather.",
                    "annotations": [],
                }
            ],
        },
        {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Berlin"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Temperature: 14 C",
        },
    ]


def _sample_poem_text() -> str:
    """Return deterministic assistant poem text."""

    return (
        "Soft rain on pine leaves\n"
        "Silver threads stitch dusk to earth\n"
        "Night drinks every sound"
    )


def _sample_reasoning_signature() -> str:
    """Return serialized provider reasoning metadata."""

    return (
        '{"id":"rs_123","type":"reasoning","summary":'
        '[{"type":"summary_text","text":"Draft a short seasonal poem."}],'
        '"encrypted_content":"enc_123","status":"completed"}'
    )

"""Serialize neutral history and tools into OpenAI Responses payloads."""

import json
from collections.abc import Sequence
from typing import TypeIs, cast

from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_image_content_param import (
    ResponseInputImageContentParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput as ResponseFunctionCallOutputParam,
)
from openai.types.responses.response_input_param import (
    ResponseFunctionCallOutputItemListParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_text_content_param import (
    ResponseInputTextContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)

from tile.providers.openai.normalized_events import Phase
from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from tile.types.stream_events import (
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)
from tile.types.tools import (
    ToolDefinition,
    ToolImageContent,
    ToolResultContent,
    ToolTextContent,
)


def serialize_history_items(
    history: Sequence[ConversationItem],
) -> ResponseInputParam:
    """Serialize replayable conversation history into OpenAI Responses items."""

    items: ResponseInputParam = []
    assistant_turn_index = 0

    for item in history:
        match item:
            case UserMessage():
                items.append(_serialize_user_message(item))
            case AssistantTurn(status="completed"):
                items.extend(
                    _serialize_assistant_turn(
                        item,
                        assistant_turn_index=assistant_turn_index,
                    )
                )
                assistant_turn_index += 1
            case ToolResultTurn():
                items.append(_serialize_tool_result_turn(item))

    return items


def serialize_tools(
    tools: Sequence[ToolDefinition],
) -> list[FunctionToolParam]:
    """Serialize app tool definitions into OpenAI Responses function tools."""

    return [_serialize_tool_definition(tool) for tool in tools]


def _serialize_user_message(
    message: UserMessage,
) -> EasyInputMessageParam:
    return {
        "role": "user",
        "content": [_build_input_text(message.content)],
    }


def _serialize_assistant_turn(
    turn: AssistantTurn,
    *,
    assistant_turn_index: int,
) -> ResponseInputParam:
    items: ResponseInputParam = []

    for block_index, block in enumerate(turn.blocks):
        match block:
            case ReasoningBlock():
                if reasoning_signature := block.metadata_string("reasoning_signature"):
                    items.append(
                        _serialize_assistant_reasoning_block(reasoning_signature)
                    )
            case TextBlock():
                items.append(
                    _serialize_assistant_text_block(
                        block,
                        assistant_turn_index=assistant_turn_index,
                        block_index=block_index,
                    )
                )
            case ToolCallBlock():
                items.append(_serialize_assistant_tool_call_block(block))

    return items


def _serialize_tool_definition(
    tool: ToolDefinition,
) -> FunctionToolParam:
    return FunctionToolParam(
        type="function",
        name=tool.name,
        description=tool.description,
        parameters={**tool.input_schema},
        strict=False,
        defer_loading=tool.defer_loading,
    )


def _serialize_assistant_reasoning_block(
    reasoning_signature: str,
) -> ResponseReasoningItemParam:
    return _deserialize_reasoning_signature(reasoning_signature)


def _serialize_assistant_text_block(
    block: TextBlock,
    *,
    assistant_turn_index: int,
    block_index: int,
) -> ResponseOutputMessageParam:
    output_text: ResponseOutputTextParam = {
        "type": "output_text",
        "text": block.text,
        "annotations": [],
    }
    message: ResponseOutputMessageParam = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": block.metadata_string("message_id")
        or _build_fallback_message_id(assistant_turn_index, block_index),
        "content": [output_text],
    }
    if phase := _read_provider_phase(block):
        message["phase"] = phase
    return message


def _serialize_assistant_tool_call_block(
    block: ToolCallBlock,
) -> ResponseFunctionToolCallParam:
    return {
        "type": "function_call",
        "id": block.metadata_string("provider_item_id") or block.call_id,
        "call_id": block.call_id,
        "name": block.name,
        "arguments": json.dumps(block.arguments),
    }


def _serialize_tool_result_turn(
    turn: ToolResultTurn,
) -> ResponseFunctionCallOutputParam:
    return {
        "type": "function_call_output",
        "call_id": turn.call_id,
        "output": _serialize_tool_result_content(turn.content),
    }


def _serialize_tool_result_content(
    content: list[ToolResultContent],
) -> str | ResponseFunctionCallOutputItemListParam:
    """Serialize provider-neutral tool result content for OpenAI Responses."""

    if _is_text_only_tool_result(content):
        return "\n".join(block.text for block in content)

    parts: ResponseFunctionCallOutputItemListParam = []
    for block in content:
        if isinstance(block, ToolImageContent):
            parts.append(_build_input_image(block))
        else:
            parts.append(_build_tool_input_text(block.text))
    return parts


def _is_text_only_tool_result(
    content: Sequence[ToolResultContent],
) -> TypeIs[Sequence[ToolTextContent]]:
    """Return whether a tool result can be replayed as plain text."""

    return all(block.type == "text" for block in content)


def _build_input_text(
    text: str,
) -> ResponseInputTextParam:
    return {
        "type": "input_text",
        "text": text,
    }


def _build_input_image(
    image: ToolImageContent,
) -> ResponseInputImageContentParam:
    """Build an OpenAI Responses image content part."""

    return {
        "type": "input_image",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
        "detail": "auto",
    }


def _build_tool_input_text(
    text: str,
) -> ResponseInputTextContentParam:
    """Build an OpenAI Responses text content part for tool outputs."""

    return {
        "type": "input_text",
        "text": text,
    }


def _build_fallback_message_id(
    assistant_turn_index: int,
    block_index: int,
) -> str:
    return f"msg_{assistant_turn_index}_{block_index}"


def _read_provider_phase(block: TextBlock) -> Phase | None:
    """Read an OpenAI message phase from provider metadata."""

    value = block.metadata_string("phase")
    if value == "commentary":
        return "commentary"
    if value == "final_answer":
        return "final_answer"
    return None


def _deserialize_reasoning_signature(
    reasoning_signature: str,
) -> ResponseReasoningItemParam:
    parsed = json.loads(reasoning_signature)
    return cast("ResponseReasoningItemParam", parsed)

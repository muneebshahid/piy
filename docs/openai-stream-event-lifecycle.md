# OpenAI Stream Event Lifecycle

This document maps raw OpenAI stream events from the SDK transport to the final agent-facing events. The executable source of truth is still the test suite:

- `tests/test_openai_provider.py` covers raw SDK payloads through provider stream events.
- `tests/test_openai_stream_assembler.py` covers normalized events through provider stream events.
- `tests/test_agent.py` covers provider stream events through agent events.

The diagrams below use actor-style Mermaid sequence diagrams. Columns are stages in the pipeline, and time flows from top to bottom.

## Actors

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant SDKA as normalize_sdk_events
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Agent attempt history

    SDK->>SDKA: OpenAI Python SDK event object
    SDKA->>Norm: transport-independent event
    Norm->>Asm: consumed in order
    Asm->>Stream: app-level stream event
    Stream->>Agent: consumed in order
    Agent->>Hist: finalized assistant/tool-result turns
```

## Stream Start And Created

`assemble_stream` emits `StreamStartEvent` when it consumes the provider `CREATED` event. The event carries the provider source and response id; assistant blocks are accumulated privately until terminal events.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseCreatedEvent
    Adapter->>Norm: CREATED(response_id)
    Norm->>Asm: CREATED
    Asm->>Stream: StreamStartEvent(source, response_id)
    Stream->>Agent: stream_start
    Agent-->>Agent: emit TurnStartEvent
    Agent-->>Agent: emit MessageStartEvent(response_id)
```

## Reasoning Item

Reasoning summary deltas and reasoning text deltas both normalize to `REASONING_DELTA` and pass through verbatim. Summary part boundaries are not surfaced as deltas, so mid-stream delta text may lack the paragraph separators present in the final summary; `REASONING_DONE` joins parts with a blank line and is the authoritative text.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=reasoning)
    Adapter->>Norm: REASONING_ADDED(item_id)
    Norm->>Asm: REASONING_ADDED
    Asm-->>Asm: append ReasoningBlock and set active_block_index
    Asm->>Stream: ReasoningStartEvent(content_index)
    Stream->>Agent: reasoning_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseReasoningSummaryTextDeltaEvent
    SDK->>Adapter: ResponseReasoningTextDeltaEvent
    Adapter->>Norm: REASONING_DELTA(delta)
    Norm->>Asm: REASONING_DELTA
    Asm-->>Asm: append delta to ReasoningBlock.summary_text
    Asm->>Stream: ReasoningDeltaEvent(content_index, delta)
    Stream->>Agent: reasoning_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseReasoningSummaryPartDoneEvent
    Adapter-->>Adapter: ignored (no normalized event)

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=reasoning)
    Adapter->>Norm: REASONING_DONE(summary_text, reasoning_signature)
    Norm->>Asm: REASONING_DONE
    Asm-->>Asm: finalize ReasoningBlock, copy block, and clear active block
    Asm->>Stream: ReasoningEndEvent(content_index, block)
    Stream->>Agent: reasoning_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Text And Refusal Item

Output-text and refusal deltas both normalize to `MESSAGE_TEXT_DELTA` and append to the active text block; the final `MESSAGE_DONE` text concatenates output-text and refusal parts.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=message)
    Adapter->>Norm: MESSAGE_ADDED(item_id, phase)
    Norm->>Asm: MESSAGE_ADDED
    Asm-->>Asm: append TextBlock and set active_block_index
    Asm->>Stream: TextStartEvent(content_index)
    Stream->>Agent: text_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseTextDeltaEvent
    Adapter->>Norm: MESSAGE_TEXT_DELTA
    Norm->>Asm: MESSAGE_TEXT_DELTA
    Asm-->>Asm: append delta to TextBlock
    Asm->>Stream: TextDeltaEvent(content_index, delta)
    Stream->>Agent: text_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseRefusalDeltaEvent
    Adapter->>Norm: MESSAGE_TEXT_DELTA
    Norm->>Asm: MESSAGE_TEXT_DELTA
    Asm-->>Asm: append delta to TextBlock
    Asm->>Stream: TextDeltaEvent(content_index, delta)
    Stream->>Agent: text_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=message)
    Adapter->>Norm: MESSAGE_DONE(text, phase)
    Norm->>Asm: MESSAGE_DONE
    Asm-->>Asm: finalize TextBlock, copy block, and clear active block
    Asm->>Stream: TextEndEvent(content_index, block)
    Stream->>Agent: text_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Tool Call Item

Argument deltas are emitted for streaming UI updates. Parsed arguments are stored on the tool-call block when the arguments-done or item-done events arrive.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=function_call)
    Adapter->>Norm: TOOL_CALL_ADDED(provider_item_id, call_id, name, arguments)
    Norm->>Asm: TOOL_CALL_ADDED
    Asm-->>Asm: append ToolCallBlock and set active_block_index
    Asm->>Stream: ToolCallStartEvent(content_index, call_id, name)
    Stream->>Agent: tool_call_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseFunctionCallArgumentsDeltaEvent
    Adapter->>Norm: TOOL_CALL_ARGUMENTS_DELTA(delta)
    Norm->>Asm: TOOL_CALL_ARGUMENTS_DELTA
    Asm->>Stream: ToolCallDeltaEvent(content_index, delta)
    Stream->>Agent: tool_call_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseFunctionCallArgumentsDoneEvent
    Adapter->>Norm: TOOL_CALL_ARGUMENTS_DONE(arguments)
    Norm->>Asm: TOOL_CALL_ARGUMENTS_DONE
    Asm-->>Asm: replace ToolCallBlock.arguments
    Note over Asm: No StreamEvent is emitted for parsed-arguments replacement.

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=function_call)
    Adapter->>Norm: TOOL_CALL_DONE(final tool call data)
    Norm->>Asm: TOOL_CALL_DONE
    Asm-->>Asm: finalize ToolCallBlock, copy block, and clear active block
    Asm->>Stream: ToolCallEndEvent(content_index, block)
    Stream->>Agent: tool_call_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Completed Turn Without Tools

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Agent attempt history

    SDK->>Adapter: ResponseCompletedEvent
    Adapter->>Norm: COMPLETED(stop_reason=stop)
    Norm->>Asm: COMPLETED
    Asm->>Stream: StreamDoneEvent(source, response_id, stop_reason, blocks)
    Stream->>Agent: stream_done
    Agent-->>Agent: build AssistantTurn.from_stream_done(...)
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent(tool_executions=[])
    Agent-->>Agent: emit AgentEndEvent
```

## Completed Turn With Tools

```mermaid
sequenceDiagram
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Tool as Tool execution
    participant Hist as Agent attempt history

    Adapter->>Norm: COMPLETED(stop_reason=tool_use)
    Norm->>Asm: COMPLETED
    Asm->>Stream: StreamDoneEvent(blocks include ToolCallBlock)
    Stream->>Agent: stream_done
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent->>Tool: execute tool call
    Agent-->>Agent: emit ToolExecutionStartEvent
    Tool-->>Agent: result
    Agent->>Hist: append ToolResultTurn to run-local history
    Agent-->>Agent: emit ToolExecutionEndEvent(outcome)
    Agent-->>Agent: emit TurnEndEvent(tool_executions=[...])
    Agent-->>Agent: request follow-up stream
    Note over Agent,Stream: The follow-up stream starts a new turn and repeats the same lifecycle.
```

## Incomplete And Failed Turns

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Agent attempt history

    SDK->>Adapter: ResponseIncompleteEvent(max_output_tokens)
    Adapter->>Norm: INCOMPLETE(stop_reason=length)
    Norm->>Asm: INCOMPLETE(length)
    Asm->>Stream: StreamDoneEvent(stop_reason=length, blocks)
    Stream->>Agent: stream_done
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent(tool_executions=[])

    SDK->>Adapter: ResponseIncompleteEvent(content_filter)
    Adapter->>Norm: INCOMPLETE(stop_reason=error)
    Norm->>Asm: INCOMPLETE(error)
    Asm->>Stream: StreamErrorEvent(error_message, blocks)
    Stream->>Agent: stream_error
    Agent-->>Agent: emit MessageEndEvent (live log only)

    SDK->>Adapter: ResponseFailedEvent or ResponseErrorEvent
    Adapter->>Norm: FAILED(message)
    Norm->>Asm: FAILED
    Asm->>Stream: StreamErrorEvent(error_message, blocks)
    Stream->>Agent: stream_error
    Agent-->>Agent: emit MessageEndEvent (live log only)
```

## Raw Event Mapping

| Raw SDK event | Normalized event | Stream assembler effect | Agent effect |
| --- | --- | --- | --- |
| `ResponseCreatedEvent` | `CREATED` | `StreamStartEvent` | `TurnStartEvent`, `MessageStartEvent(response_id)` |
| `ResponseOutputItemAddedEvent` with reasoning item | `REASONING_ADDED` | `ReasoningStartEvent` | `MessageUpdateEvent` |
| `ResponseReasoningSummaryTextDeltaEvent` | `REASONING_DELTA` | `ReasoningDeltaEvent` | `MessageUpdateEvent` |
| `ResponseReasoningTextDeltaEvent` | `REASONING_DELTA` | `ReasoningDeltaEvent` | `MessageUpdateEvent` |
| `ResponseReasoningSummaryPartDoneEvent` | ignored | — | — |
| `ResponseOutputItemDoneEvent` with reasoning item | `REASONING_DONE` | `ReasoningEndEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemAddedEvent` with message item | `MESSAGE_ADDED` | Starts the text block | `TextStartEvent` |
| `ResponseTextDeltaEvent` | `MESSAGE_TEXT_DELTA` | `TextDeltaEvent` | `MessageUpdateEvent` |
| `ResponseRefusalDeltaEvent` | `MESSAGE_TEXT_DELTA` | `TextDeltaEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemDoneEvent` with message item | `MESSAGE_DONE` | `TextEndEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemAddedEvent` with function-call item | `TOOL_CALL_ADDED` | `ToolCallStartEvent` | `MessageUpdateEvent` |
| `ResponseFunctionCallArgumentsDeltaEvent` | `TOOL_CALL_ARGUMENTS_DELTA` | `ToolCallDeltaEvent` | `MessageUpdateEvent` |
| `ResponseFunctionCallArgumentsDoneEvent` | `TOOL_CALL_ARGUMENTS_DONE` | Replaces parsed arguments; no new stream event | No direct event |
| `ResponseOutputItemDoneEvent` with function-call item | `TOOL_CALL_DONE` | `ToolCallEndEvent` | `MessageUpdateEvent` |
| `ResponseCompletedEvent` | `COMPLETED` | `StreamDoneEvent` | `MessageEndEvent`, `TurnEndEvent`, optional tool execution |
| `ResponseIncompleteEvent` with length stop | `INCOMPLETE(length)` | `StreamDoneEvent` | `MessageEndEvent`, `TurnEndEvent` |
| `ResponseIncompleteEvent` with content-filter stop | `INCOMPLETE(error)` | `StreamErrorEvent` | Error-bearing `MessageEndEvent`; `RunEndEvent` terminates the open turn and agent |
| `ResponseFailedEvent` or `ResponseErrorEvent` | `FAILED` | `StreamErrorEvent` | Error-bearing `MessageEndEvent`; `RunEndEvent` terminates the open turn and agent |

## Run Lifecycle

Every run log begins with `RunStartEvent` and ends with one terminal event.
`RunEndEvent(outcome)` closes a durably finalized run; `RunFaultEvent` closes
the local stream when that durable ending cannot be established. Inner events
carry no such guarantee. A failure or cancellation can leave inner starts
without their normal ends; the terminal run event closes every scope still
open. Hard process death is outside this contract.

| Scope | Start | Normal end |
| --- | --- | --- |
| run | `RunStartEvent` (guaranteed first) | `RunEndEvent(outcome)` or `RunFaultEvent` (guaranteed last) |
| agent attempt | `AgentStartEvent` | `AgentEndEvent` |
| turn | `TurnStartEvent` | `TurnEndEvent(assistant_turn, tool_executions)` |
| message | `MessageStartEvent` | `MessageEndEvent(assistant_turn)` |
| tool execution | `ToolExecutionStartEvent(call_id)` | `ToolExecutionEndEvent(outcome)` |

Rules:

- `RunExecution` emits `RunStartEvent` before execution starts, so
  every run log begins with a run start on every path, including an
  abort that lands before the first tick.
- `RunEndEvent` is emitted only after harness-owned durable finalization.
  Execution returns a candidate `RunOutcome`; `RunExecution` persists it and
  then emits the matching run end. A finalization failure emits
  `RunFaultEvent` instead.
- A tool execution that *fails* still gets its `ToolExecutionEndEvent`:
  the executor boundary wraps every tool failure into an error outcome.
  Cancellation or infrastructure failure while the tool is active leaves
  the tool scope open until `RunEndEvent`.
- An in-band provider `StreamErrorEvent` is a terminal message, so its
  partial error-bearing assistant turn is retained in `MessageEndEvent`.
  It is not committed to replayable history. The failed turn and agent remain
  open until the failed `RunEndEvent`.
- Typed-result runs close each agent attempt before the follow-up message
  or next attempt starts; the final `AgentEndEvent` precedes `RunEndEvent`.
  Agent events carry no attempt label: attempts are strictly sequential,
  so position in the log identifies them, with `ResultFollowUpEvent`
  separating retries.
- The terminal run record and replayable history are committed atomically
  before `RunEndEvent` closes the live log. If that transaction fails,
  `RunHandle.wait()` returns `Faulted` and `RunFaultEvent` closes the local
  event stream. A stale writer reconciles to the authoritative stored outcome.
- Provider stream fragments (`TextStart/End`, `ReasoningStart/End`,
  `ToolCallStart/Delta/End`, and the provider `StreamStart/Done/Error`
  events) are message content, not lifecycle scopes; the containing
  message end or terminal run end ends their outstanding state.
Abort ordering during a tool call leaves inner scopes open for the run end:

```text
...
MessageEnd(assistant_turn)
ToolExecutionStart(call_id)
RunEnd(outcome=Aborted)
```

Typed-result follow-up ordering:

```text
RunStart
  AgentStart
    ...
  AgentEnd
  ResultFollowUp
  AgentStart
    ...
  AgentEnd
RunEnd(outcome)
```

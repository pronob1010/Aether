"""Anthropic (Claude) provider adapter for the Messages API.

Notable shape differences from the OpenAI/Gemini adapters:
  - `max_tokens` is *required* by the Messages API, so the provider carries a
    `default_max_tokens`.
  - System messages go on the top-level `system` param, not inline in the
    conversation (like Gemini).
  - `temperature` is intentionally NOT forwarded: current Claude models
    (Opus 4.7+ and Fable) reject sampling params with a 400, so steering is
    left to prompting. Honoring `request.temperature` would break the default
    model.
  - Tool calls arrive as `tool_use` content blocks; results go back as
    `tool_result` blocks wrapped in a user turn.

The `anthropic` SDK is imported lazily in `__init__` so this module (and its
pure translation helpers) can be imported and tested without the SDK present.
"""
from collections.abc import AsyncIterator
from typing import Any

from agartha.llm.contracts import (
    DocumentPart,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    TextPart,
    ToolCall,
    text_of,
)
from agartha.tools import get_tool


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Pull system turns out of the conversation onto the top-level param."""
    system_parts: list[str] = []
    conversation: list[Message] = []
    for msg in messages:
        if msg.role == "system" and msg.content:
            system_parts.append(text_of(msg.content))
        else:
            conversation.append(msg)
    return ("\n".join(system_parts) if system_parts else None, conversation)


def _anthropic_content(content: Any) -> Any:
    """Translate a Message's content into Anthropic's content format.

    Plain strings pass through unchanged; a list of parts becomes the
    content-block array (text + image + document source blocks).
    """
    if not isinstance(content, list):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": part.data,
                },
            })
        elif isinstance(part, DocumentPart):
            blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": part.data,
                },
            })
    return blocks


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Agartha's Message list into Anthropic's message format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            out.append({"role": "assistant", "content": blocks})
        elif msg.role == "tool" and msg.tool_call_id is not None:
            # Tool results are delivered as a user turn carrying a tool_result.
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }],
            })
        elif msg.role in ("user", "assistant") and msg.content is not None:
            out.append({"role": msg.role, "content": _anthropic_content(msg.content)})
    return out


def _tools_payload(tool_names: list[str] | None) -> list[dict[str, Any]] | None:
    if not tool_names:
        return None
    payload: list[dict[str, Any]] = []
    for name in tool_names:
        spec = get_tool(name)
        payload.append({
            "name": spec.schema["name"],
            "description": spec.schema.get("description", ""),
            "input_schema": spec.schema["parameters"],
        })
    return payload


def _parse_tool_calls(content: Any) -> list[ToolCall]:
    """Extract tool_use blocks from a response's content list."""
    parsed: list[ToolCall] = []
    for block in content or []:
        if getattr(block, "type", None) == "tool_use":
            parsed.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=dict(block.input) if block.input else {},
            ))
    return parsed


def _extract_text(content: Any) -> str:
    """Concatenate the text blocks of a response's content list."""
    return "".join(
        block.text
        for block in (content or [])
        if getattr(block, "type", None) == "text"
    )


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-opus-4-8",
        default_max_tokens: int = 4096,
        *,
        client: Any | None = None,
    ):
        # `client` is an injection seam for testing — pass a fake to exercise
        # complete()/stream() without the SDK or network. Production callers
        # pass only api_key and let the SDK client be constructed here.
        if client is not None:
            self.client = client
        else:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens

    def _base_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        system, conversation = _split_system(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "max_tokens": self.default_max_tokens,
            "messages": _to_anthropic_messages(conversation),
        }
        if system is not None:
            kwargs["system"] = system
        if tools := _tools_payload(request.tools):
            kwargs["tools"] = tools
        return kwargs

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self.client.messages.create(**self._base_kwargs(request))
        return LLMResponse(
            text=_extract_text(response.content),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=_parse_tool_calls(response.content),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream text deltas and tool-use blocks.

        Claude streams a tool call as a `content_block_start` (carrying id and
        name) followed by `input_json_delta` events whose `partial_json` must
        be accumulated, then a `content_block_stop`. Token counts arrive split
        across `message_start` (input) and `message_delta` (output); the final
        consolidated tool-call chunk carries both.
        """
        import json

        kwargs = self._base_kwargs(request)
        model = kwargs["model"]

        # Per-content-block accumulators, keyed by the streamed block index.
        tool_blocks: dict[int, dict[str, str]] = {}
        input_tokens: int | None = None
        output_tokens: int | None = None

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "message_start":
                    usage = getattr(event.message, "usage", None)
                    if usage is not None:
                        input_tokens = usage.input_tokens

                elif etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        tool_blocks[event.index] = {
                            "id": block.id, "name": block.name, "args": "",
                        }

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        yield LLMStreamChunk(text=delta.text, model=model)
                    elif dtype == "input_json_delta" and event.index in tool_blocks:
                        tool_blocks[event.index]["args"] += delta.partial_json

                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None and usage.output_tokens is not None:
                        output_tokens = usage.output_tokens

        # Emit a final chunk carrying any tool calls plus the token counts.
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_blocks):
            buf = tool_blocks[idx]
            try:
                args = json.loads(buf["args"]) if buf["args"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=buf["id"], name=buf["name"], arguments=args))

        yield LLMStreamChunk(
            text="",
            model=model,
            finish_reason="tool_calls" if tool_calls else "stop",
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

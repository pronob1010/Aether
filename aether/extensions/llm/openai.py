import json
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from aether.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
)
from aether.tools import get_tool


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Aether's Message list to OpenAI's chat-completions format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.role == "assistant" and msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.role == "tool" and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        out.append(d)
    return out


def _tools_payload(tool_names: list[str] | None) -> list[dict[str, Any]] | None:
    """Look up each tool's auto-generated schema and wrap for OpenAI."""
    if not tool_names:
        return None
    return [
        {"type": "function", "function": get_tool(name).schema}
        for name in tool_names
    ]


def _parse_tool_calls(msg: Any) -> list[ToolCall]:
    raw = getattr(msg, "tool_calls", None) or []
    parsed: list[ToolCall] = []
    for tc in raw:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        parsed.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return parsed


class OpenAIProvider:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": _to_openai_messages(request.messages),
            "temperature": request.temperature,
        }
        if tools := _tools_payload(request.tools):
            kwargs["tools"] = tools
        completion = await self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        return LLMResponse(
            text=message.content or "",
            model=completion.model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            tool_calls=_parse_tool_calls(message),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        # NOTE: tool calling during streaming is not supported in Phase C.
        # OpenAI delivers tool_call args incrementally across chunks; correctly
        # buffering them while still emitting text deltas needs a small state
        # machine. Tracked for a future phase.
        stream = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=_to_openai_messages(request.messages),
            temperature=request.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    yield LLMStreamChunk(
                        text="",
                        model=chunk.model,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                continue
            choice = chunk.choices[0]
            yield LLMStreamChunk(
                text=choice.delta.content or "",
                model=chunk.model,
                finish_reason=choice.finish_reason,
            )

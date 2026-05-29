from typing import AsyncIterator
from openai import AsyncOpenAI
from aether.llm.contracts import LLMRequest, LLMResponse, LLMStreamChunk

class OpenAIProvider:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
        )
        return LLMResponse(
            text=model.choices[0].message.content or "",
            model=model.model,
            input_tokens=model.usage.prompt_tokens,
            output_tokens=model.usage.completion_tokens,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        stream = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            # Final usage-only chunk has no choices.
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

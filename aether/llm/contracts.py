from typing import AsyncIterator, Protocol, runtime_checkable
from pydantic import BaseModel

class LLMRequest(BaseModel):
    prompt: str
    model: str | None = None
    temperature: float = 0.7

class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int

class LLMStreamChunk(BaseModel):
    """One delta in a streaming response.

    `text` is the new tokens since the previous chunk (a delta, not cumulative).
    Provider metadata (`model`, `finish_reason`, token counts) is populated
    when the underlying SDK emits it — typically only on the final chunk.
    """
    text: str = ""
    model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        ...

import base64
import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool invocation the LLM wants the framework to execute.

    `id` is provider-assigned; we echo it back when delivering the tool
    result so the LLM can match call → result.
    """
    id: str
    name: str
    arguments: dict[str, Any] = {}


def _read_base64(path: str | Path) -> tuple[str, str]:
    """Read a file and return (media_type, base64_data).

    Media type is guessed from the filename extension, falling back to a
    generic binary type when unknown.
    """
    p = Path(path)
    media_type, _ = mimetypes.guess_type(p.name)
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return (media_type or "application/octet-stream", data)


class TextPart(BaseModel):
    """A run of text within a multi-part message turn."""
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image attached to a turn, carried inline as base64."""
    type: Literal["image"] = "image"
    media_type: str          # e.g. "image/png", "image/jpeg"
    data: str                # base64-encoded bytes

    @classmethod
    def from_path(cls, path: str | Path) -> "ImagePart":
        media_type, data = _read_base64(path)
        return cls(media_type=media_type, data=data)


class DocumentPart(BaseModel):
    """A document (e.g. a PDF) attached to a turn, carried inline as base64."""
    type: Literal["document"] = "document"
    media_type: str          # e.g. "application/pdf"
    data: str                # base64-encoded bytes

    @classmethod
    def from_path(cls, path: str | Path) -> "DocumentPart":
        media_type, data = _read_base64(path)
        return cls(media_type=media_type, data=data)


# A single piece of message content. The `type` field discriminates the union
# so Pydantic round-trips it losslessly through session storage.
ContentPart = Annotated[
    TextPart | ImagePart | DocumentPart,
    Field(discriminator="type"),
]


def text_of(content: "str | list[ContentPart] | None") -> str:
    """Best-effort plain-text view of a message's content.

    Returns a string unchanged, joins the text parts of a multi-part list, and
    maps None to "". Used where a component only cares about the text (system
    handling, token estimates) and should ignore attached media.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, TextPart))


class Message(BaseModel):
    """One turn in a conversation.

    Roles:
      - 'system'    instructions / persona
      - 'user'      user input
      - 'assistant' LLM output (may include tool_calls)
      - 'tool'      result from executing a tool the LLM asked for

    `content` is either a plain string (the common case) or a list of typed
    parts (text + images + documents) for multimodal turns. Provider adapters
    translate the parts into each backend's native format.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] = []         # only on assistant turns
    tool_call_id: str | None = None         # only on tool turns


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    tools: list[str] | None = None          # populated in Phase C


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall] = []         # populated in Phase C


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
    tool_calls: list[ToolCall] = []         # populated in Phase C


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        ...

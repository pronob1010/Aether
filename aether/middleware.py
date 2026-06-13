"""Response middleware — user-authored layers over the model's final answer.

A `ResponseMiddleware` is called with the request and the model's final
`LLMResponse` and returns a (possibly modified) response. Use it to redact,
annotate, enforce policy, or — as with `GroundingGuard` — gate answers that
aren't supported by a known set of source documents.

Middleware runs only on the *final* answer returned to the caller, not on the
intermediate tool-calling rounds, and only on `complete()` / `ask()` (not on
streaming). Configure via `Aether(response_middleware=[...])`.

This complements the EventBus: events *observe* the request path; middleware
*shapes* the result.
"""
import inspect
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from aether.llm.contracts import LLMRequest, LLMResponse


@runtime_checkable
class ResponseMiddleware(Protocol):
    async def process(
        self, request: LLMRequest, response: LLMResponse
    ) -> LLMResponse:
        """Return the response unchanged, a modified copy, or raise to reject.

        May be implemented sync or async — the client awaits the result only
        when it is awaitable.
        """
        ...


# Verifier signature for GroundingGuard: given the answer text and the source
# documents, return (optionally via awaitable) whether the answer is grounded.
Verifier = Callable[[str, list[str]], bool | Awaitable[bool]]


DEFAULT_REFUSAL = (
    "I can only answer using the provided documents, and I couldn't find "
    "support for that there."
)

# Tiny stopword set so the default overlap heuristic isn't trivially satisfied
# by filler words that appear in any text.
_STOPWORDS = frozenset(
    "a an the and or but if then is are was were be been being of to in on for "
    "with as by at from this that these those it its i you he she they we".split()
)


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    }


def _overlap_grounded(answer: str, sources: list[str], threshold: float) -> bool:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return True  # nothing substantive to ground
    source_tokens: set[str] = set()
    for s in sources:
        source_tokens |= _tokens(s)
    overlap = answer_tokens & source_tokens
    return len(overlap) / len(answer_tokens) >= threshold


class GroundingGuard:
    """Reject answers not supported by a known set of source documents.

    For RAG: pass the retrieved chunks as `sources`. If the model's answer
    isn't grounded in them, the answer text is replaced with `refusal` (tool
    calls are cleared too).

    The default check is a transparent token-overlap heuristic — adequate for
    catching blatant fabrication, but for production pass your own `verifier`
    (e.g. an LLM judge); it may be sync or async:

        async def judge(answer, sources): ...    # call your model here
        Aether(response_middleware=[GroundingGuard(chunks, verifier=judge)])
    """

    def __init__(
        self,
        sources: list[str],
        *,
        verifier: Verifier | None = None,
        refusal: str = DEFAULT_REFUSAL,
        threshold: float = 0.5,
    ):
        self.sources = list(sources)
        self.verifier = verifier
        self.refusal = refusal
        self.threshold = threshold

    async def process(
        self, request: LLMRequest, response: LLMResponse
    ) -> LLMResponse:
        text = response.text.strip()
        if not text:
            return response

        if self.verifier is not None:
            verdict = self.verifier(text, self.sources)
            if inspect.isawaitable(verdict):
                verdict = await verdict
            grounded = bool(verdict)
        else:
            grounded = _overlap_grounded(text, self.sources, self.threshold)

        if grounded:
            return response
        return response.model_copy(update={"text": self.refusal, "tool_calls": []})

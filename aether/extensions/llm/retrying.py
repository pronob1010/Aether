import logging
from typing import AsyncIterator, Type
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from aether.llm.contracts import LLMProvider, LLMRequest, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)

class RetryingProvider:
    """
    A decorator provider that adds resilience to any LLMProvider.
    Uses exponential backoff with jitter to retry transient errors.

    Streaming behavior: retry applies ONLY to opening the stream and
    receiving the first chunk. Once any chunk has been yielded to the
    caller, further errors propagate — re-streaming would duplicate
    output the caller has already seen.
    """
    def __init__(
        self,
        inner_provider: LLMProvider,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
        retry_exceptions: tuple[Type[Exception], ...] = (Exception,)
    ):
        self.inner_provider = inner_provider
        self.retry_logic = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=min_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        async for attempt in self.retry_logic:
            with attempt:
                return await self.inner_provider.complete(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        # Retry the handshake: open the stream + pull the first chunk.
        # After the first chunk yields, errors propagate as-is.
        iterator: AsyncIterator[LLMStreamChunk] | None = None
        first_chunk: LLMStreamChunk | None = None

        async for attempt in self.retry_logic:
            with attempt:
                iterator = aiter(self.inner_provider.stream(request))
                first_chunk = await anext(iterator)

        # Type narrowing: tenacity guarantees these are set on success.
        assert iterator is not None and first_chunk is not None

        yield first_chunk
        async for chunk in iterator:
            yield chunk

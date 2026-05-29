"""Cost-tracking decorator + the stats objects it exposes.

Sits outermost in the decorator stack so it only counts what actually
billed (successful final outcome, not failed retries).
"""
from dataclasses import dataclass, field
from typing import AsyncIterator
from pydantic import BaseModel
from aether.llm.contracts import LLMProvider, LLMRequest, LLMResponse, LLMStreamChunk


class ModelPricing(BaseModel):
    """Cost per million tokens for one model, separately for input and output."""
    input_per_1m: float
    output_per_1m: float


# Defaults as of early 2026 — refresh from provider pricing pages.
# Users override via CostTrackingConfig.pricing.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-3.5-turbo":     ModelPricing(input_per_1m=0.50, output_per_1m=1.50),
    "gpt-4o":            ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini":       ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
    "gemini-2.0-flash":  ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
    "gemini-2.5-flash":  ModelPricing(input_per_1m=0.30, output_per_1m=2.50),
    "fake-model":        ModelPricing(input_per_1m=0.0, output_per_1m=0.0),
}


@dataclass
class TokenUsage:
    """Cumulative usage for one model."""
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1


@dataclass
class UsageStats:
    """Aggregated stats across all calls a CostTrackingProvider has observed."""
    by_model: dict[str, TokenUsage] = field(default_factory=dict)
    pricing: dict[str, ModelPricing] = field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.by_model.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.by_model.values())

    @property
    def total_requests(self) -> int:
        return sum(u.requests for u in self.by_model.values())

    @property
    def total_cost_usd(self) -> float | None:
        """Total cost in USD if any of the observed models have pricing data.

        Returns None if no pricing is configured. Models without pricing are
        excluded from the sum — better to under-report than to invent numbers.
        """
        if not self.pricing:
            return None
        total = 0.0
        for model, usage in self.by_model.items():
            price = self.pricing.get(model)
            if price is None:
                continue
            total += usage.input_tokens / 1_000_000 * price.input_per_1m
            total += usage.output_tokens / 1_000_000 * price.output_per_1m
        return total


class CostTrackingProvider:
    """Decorator that records token usage on every call. Pass-through otherwise.

    Position in the decorator stack matters: this should sit OUTERMOST so it
    only counts what actually billed (i.e., the final successful outcome
    after retries, not each individual retry attempt).
    """

    def __init__(
        self,
        inner_provider: LLMProvider,
        pricing: dict[str, ModelPricing] | None = None,
    ):
        self.inner_provider = inner_provider
        self.stats = UsageStats(pricing=pricing if pricing is not None else {})

    def _record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        if model not in self.stats.by_model:
            self.stats.by_model[model] = TokenUsage()
        self.stats.by_model[model].add(input_tokens, output_tokens)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self.inner_provider.complete(request)
        self._record(response.model, response.input_tokens, response.output_tokens)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        # Capture token counts from the last chunk(s) that carry them.
        # Only record AFTER the stream completes normally — an aborted stream
        # wasn't billed (matches OpenAI/Gemini behavior).
        final_input: int | None = None
        final_output: int | None = None
        final_model: str | None = None

        async for chunk in self.inner_provider.stream(request):
            if chunk.input_tokens is not None:
                final_input = chunk.input_tokens
            if chunk.output_tokens is not None:
                final_output = chunk.output_tokens
            if chunk.model is not None:
                final_model = chunk.model
            yield chunk

        if final_input is not None and final_output is not None and final_model is not None:
            self._record(final_model, final_input, final_output)

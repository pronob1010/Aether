from pydantic import BaseModel
from aether.llm.contracts import LLMProvider
from aether.extensions.llm.cost_tracking import ModelPricing
from aether.extensions.llm.factory import make_provider


class RetryConfig(BaseModel):
    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 10.0


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    recovery_timeout: float = 30.0


class CostTrackingConfig(BaseModel):
    """Toggle cost tracking. Set `pricing` to override the built-in defaults
    (e.g., for models we don't ship pricing for, or to use your enterprise rates).
    """
    pricing: dict[str, ModelPricing] | None = None


class ProviderConfig(BaseModel):
    name: str
    api_key: str | None = None
    default_model: str | None = None
    retry: RetryConfig | None = None
    circuit_breaker: CircuitBreakerConfig | None = None
    cost_tracking: CostTrackingConfig | None = None


def build_provider(config: ProviderConfig) -> LLMProvider:
    kwargs = {}
    if config.api_key:       kwargs["api_key"] = config.api_key
    if config.default_model: kwargs["default_model"] = config.default_model
    provider = make_provider(config.name, **kwargs)

    # Order is load-bearing:
    #   1. Retry sits INSIDE the breaker so one retry-exhausted call counts
    #      as ONE breaker failure, not N.
    #   2. Cost tracking sits OUTERMOST so it only counts what actually
    #      billed (the final outcome, not each retry attempt).
    if config.retry:
        from aether.extensions.llm.retrying import RetryingProvider
        provider = RetryingProvider(provider, **config.retry.model_dump())
    if config.circuit_breaker:
        from aether.extensions.llm.circuit_breaker import CircuitBreakerProvider
        provider = CircuitBreakerProvider(provider, **config.circuit_breaker.model_dump())
    if config.cost_tracking:
        from aether.extensions.llm.cost_tracking import (
            CostTrackingProvider,
            DEFAULT_PRICING,
        )
        pricing = config.cost_tracking.pricing or DEFAULT_PRICING
        provider = CostTrackingProvider(provider, pricing=pricing)

    return provider

"""LLM provider implementations and the LLM-specific registration helper.

Public surface:
  - `register_provider` — decorator for third-party providers
  - `make_provider`     — name → instance lookup
  - `build_provider`    — config → composed (provider + decorators) stack
  - `ProviderConfig`, `RetryConfig`, `CircuitBreakerConfig`,
    `CostTrackingConfig` — config models
  - `UsageStats`, `TokenUsage`, `ModelPricing`, `DEFAULT_PRICING` —
    cost-tracking primitives

Concrete adapters (`OpenAIProvider`, `GeminiProvider`, `FakeProvider`) and
decorators (`RetryingProvider`, `CircuitBreakerProvider`,
`CostTrackingProvider`) live as submodules and are imported lazily.
"""

from aether.extensions.llm.registry import register_provider, LLM_PROVIDER_KIND
from aether.extensions.llm.factory import make_provider
from aether.extensions.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    CostTrackingConfig,
    build_provider,
)
from aether.extensions.llm.cost_tracking import (
    UsageStats,
    TokenUsage,
    ModelPricing,
    DEFAULT_PRICING,
)

__all__ = [
    "register_provider",
    "LLM_PROVIDER_KIND",
    "make_provider",
    "build_provider",
    "ProviderConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
    "CostTrackingConfig",
    "UsageStats",
    "TokenUsage",
    "ModelPricing",
    "DEFAULT_PRICING",
]

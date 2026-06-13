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

from aether.extensions.llm.builder import (
    CircuitBreakerConfig,
    CostTrackingConfig,
    ProviderConfig,
    RetryConfig,
    build_provider,
)
from aether.extensions.llm.cost_tracking import (
    DEFAULT_PRICING,
    ModelPricing,
    TokenUsage,
    UsageStats,
)
from aether.extensions.llm.factory import make_provider
from aether.extensions.llm.registry import LLM_PROVIDER_KIND, register_provider

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

"""All extensions to Agartha's pluggable subsystems.

Each subdirectory corresponds to a `kind` in the generic registry:
  - `agartha.extensions.llm` — LLM provider adapters + resilience decorators
  - `agartha.extensions.vector` — (future) vector store implementations
  - `agartha.extensions.database` — (future) database implementations

Extension authors add to this namespace via decorators like `@register_provider`.
"""

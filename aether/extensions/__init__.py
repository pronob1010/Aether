"""All extensions to Aether's pluggable subsystems.

Each subdirectory corresponds to a `kind` in the generic registry:
  - `aether.extensions.llm` — LLM provider adapters + resilience decorators
  - `aether.extensions.vector` — (future) vector store implementations
  - `aether.extensions.database` — (future) database implementations

Extension authors add to this namespace via decorators like `@register_provider`.
"""

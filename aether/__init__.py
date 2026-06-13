"""
Aether — a Python framework for building AI-native applications.

Blends classical software engineering patterns with the demands of
LLM-based execution: provider abstraction, prompt pipelines, tool
registries, memory, context budgeting, reasoning strategies, and
observability.
"""

from aether.client import Aether
from aether.llm import LLMProvider, LLMRequest, LLMResponse, Message, ToolCall, ask
from aether.llm.contracts import TextPart, ImagePart, DocumentPart
from aether.extensions.llm.registry import register_provider
from aether.extensions.llm.cost_tracking import UsageStats, TokenUsage, ModelPricing
from aether.registry import register, register_lazy
from aether.tools import register_tool, list_tools, get_tool
from aether.events import EventBus
from aether.memory import Session, SessionStore
from aether.middleware import ResponseMiddleware, GroundingGuard

__all__ = [
    # Entry point
    "Aether",
    # Conversation primitives
    "Message",
    "ToolCall",
    # Multimodal message content (documents / images on a turn)
    "TextPart",
    "ImagePart",
    "DocumentPart",
    # Response middleware ("result layers")
    "ResponseMiddleware",
    "GroundingGuard",
    # Sessions (stateful conversations)
    "Session",
    "SessionStore",
    # Cost tracking primitives (exposed via Aether.usage)
    "UsageStats",
    "TokenUsage",
    "ModelPricing",
    # Observability (event constants + payload dataclasses live in aether.events)
    "EventBus",
    # Generic extension API (any subsystem: LLM, vector store, DB, ...)
    "register",
    "register_lazy",
    # LLM-specific convenience
    "register_provider",
    # Tool registration
    "register_tool",
    "list_tools",
    "get_tool",
    # LLM contracts
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ask",
]

__version__ = "0.0.1"

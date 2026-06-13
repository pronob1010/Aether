"""
Agartha — a Python framework for building AI-native applications.

Blends classical software engineering patterns with the demands of
LLM-based execution: provider abstraction, prompt pipelines, tool
registries, memory, context budgeting, reasoning strategies, and
observability.
"""

from agartha.client import Agartha
from agartha.agent import Agent
from agartha.llm import LLMProvider, LLMRequest, LLMResponse, Message, ToolCall, ask
from agartha.llm.contracts import TextPart, ImagePart, DocumentPart
from agartha.extensions.llm.registry import register_provider
from agartha.extensions.llm.cost_tracking import UsageStats, TokenUsage, ModelPricing
from agartha.registry import register, register_lazy
from agartha.tools import register_tool, list_tools, get_tool
from agartha.events import EventBus
from agartha.memory import Session, SessionStore
from agartha.middleware import Middleware, GroundingGuard

__all__ = [
    # Entry point
    "Agartha",
    # Declarative agent runtime (+ sub-agent delegation via .as_tool())
    "Agent",
    # Conversation primitives
    "Message",
    "ToolCall",
    # Multimodal message content (documents / images on a turn)
    "TextPart",
    "ImagePart",
    "DocumentPart",
    # Middleware pipeline (before_request / on_tool_call / after_response)
    "Middleware",
    "GroundingGuard",
    # Sessions (stateful conversations)
    "Session",
    "SessionStore",
    # Cost tracking primitives (exposed via Agartha.usage)
    "UsageStats",
    "TokenUsage",
    "ModelPricing",
    # Observability (event constants + payload dataclasses live in agartha.events)
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

"""
Aether — a Python framework for building AI-native applications.

Blends classical software engineering patterns with the demands of
LLM-based execution: provider abstraction, prompt pipelines, tool
registries, memory, context budgeting, reasoning strategies, and
observability.
"""

from aether.agent import Agent
from aether.client import Aether
from aether.events import EventBus
from aether.extensions.llm.cost_tracking import ModelPricing, TokenUsage, UsageStats
from aether.extensions.llm.registry import register_provider
from aether.llm import LLMProvider, LLMRequest, LLMResponse, Message, ToolCall, ask
from aether.llm.contracts import DocumentPart, ImagePart, TextPart
from aether.memory import Session, SessionStore
from aether.middleware import GroundingGuard, Middleware
from aether.registry import register, register_lazy
from aether.tools import get_tool, list_tools, register_tool

__all__ = [
    # Entry point
    "Aether",
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

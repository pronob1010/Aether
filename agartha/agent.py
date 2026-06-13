"""Agent — a declarative, reusable bundle of model + instructions + tools.

`Agartha` is the low-level client you *call*; an `Agent` is a configuration you
*declare* once and run many times, letting the framework own the loop:

    research = Agent(
        "researcher",
        instructions="You answer questions using the available tools.",
        tools=["http_get", "get_current_time"],
        provider=my_provider,
    )
    answer = await research.run_text("What time is it in Tokyo?")

Agents compose: `agent.as_tool()` registers the agent as a tool that *another*
agent can call, so a coordinator can delegate sub-tasks. The delegated run is an
ordinary tool call inside the parent's loop, so it reuses tool dispatch, the
middleware pipeline, events, and cost tracking for free.
"""
from contextvars import ContextVar

from agartha.client import Agartha
from agartha.config import get_max_delegation_depth
from agartha.llm.contracts import LLMProvider, LLMResponse, Message
from agartha.middleware import Middleware
from agartha.tools import register_tool

# Current sub-agent delegation depth. A ContextVar so the value propagates down
# the nested await chain (parent run -> delegate tool -> child run) and is
# copied — and thus isolated — into each branch of a parallel fan-out.
_delegation_depth: ContextVar[int] = ContextVar("agartha_delegation_depth", default=0)


class Agent:
    """A named, reusable agent: instructions + tools + model over a client.

    Provide a ready `client`, a bare `provider` (a client is built around it),
    or neither (a client is auto-configured from the environment). `middleware`
    is applied to the client built here; pass a `client` directly if you want
    to control its middleware yourself.
    """

    def __init__(
        self,
        name: str,
        *,
        instructions: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        client: Agartha | None = None,
        provider: LLMProvider | None = None,
        middleware: list[Middleware] | None = None,
    ):
        if client is not None and provider is not None:
            raise ValueError("Pass either `client` or `provider`, not both.")
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.model = model
        if client is not None:
            self.client = client
        elif provider is not None:
            self.client = Agartha(provider, middleware=middleware)
        else:
            self.client = Agartha(middleware=middleware)

    def _build_messages(self, prompt: str | list[Message]) -> list[Message]:
        messages: list[Message] = []
        if self.instructions:
            messages.append(Message(role="system", content=self.instructions))
        if isinstance(prompt, str):
            messages.append(Message(role="user", content=prompt))
        else:
            messages.extend(prompt)
        return messages

    async def run(self, prompt: str | list[Message]) -> LLMResponse:
        """Run the agent over a prompt and return the full response.

        Prepends the agent's `instructions` as a system turn and drives the
        client's tool loop with the agent's `tools` and `model`.
        """
        return await self.client.complete(
            self._build_messages(prompt),
            tools=self.tools,
            model=self.model,
        )

    async def run_text(self, prompt: str | list[Message]) -> str:
        """Run the agent and return just the answer text."""
        return (await self.run(prompt)).text

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        max_depth: int | None = None,
    ) -> str:
        """Register this agent as a tool another agent can call, and return its
        tool name.

        The returned name goes in a coordinator's `tools=[...]`; when the
        coordinator calls it, the parent loop dispatches a fresh run of this
        agent with the given task and feeds the answer back as the tool result.
        When a coordinator emits several delegation calls in one turn, the
        client dispatches them concurrently (see `Agartha.complete`).

        Delegation is bounded: once `max_depth` nested levels are reached
        (default from `AGARTHA_MAX_DELEGATION_DEPTH`), a further delegation is
        refused with an error string instead of recursing without bound.
        """
        tool_name = name or self.name
        tool_description = (
            description
            or self.instructions
            or f"Delegate a sub-task to the {self.name!r} agent and get its answer."
        )

        @register_tool(name=tool_name, description=tool_description)
        async def _delegate(task: str) -> str:
            """Delegate a task to a sub-agent.

            Args:
                task: A self-contained description of the sub-task to perform.
            """
            limit = max_depth if max_depth is not None else get_max_delegation_depth()
            depth = _delegation_depth.get()
            if depth >= limit:
                return (
                    f"Error: delegation depth limit ({limit}) reached; refusing "
                    f"to delegate to {self.name!r} to avoid runaway recursion."
                )
            token = _delegation_depth.set(depth + 1)
            try:
                return await self.run_text(task)
            finally:
                _delegation_depth.reset(token)

        return tool_name

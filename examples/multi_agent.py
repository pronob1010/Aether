"""Multi-agent: a coordinator delegates to specialist sub-agents.

`agent.as_tool()` exposes an agent as a tool another agent can call. The
coordinator decides when to delegate; each delegated run is an ordinary tool
call in the coordinator's loop (so it reuses dispatch, middleware, events, and
cost tracking). When the coordinator emits several delegations in one turn they
run concurrently, and nested delegation is depth-bounded.

    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/multi_agent.py
"""
import asyncio

from aether import Agent


async def main() -> str:
    researcher = Agent(
        "researcher",
        instructions="Research the question and report concise, factual notes.",
    )
    summarizer = Agent(
        "summarizer",
        instructions="Summarize the material you are given in two sentences.",
    )
    coordinator = Agent(
        "coordinator",
        instructions=(
            "Delegate research to the researcher, then ask the summarizer to "
            "condense the findings into a short brief."
        ),
        tools=[researcher.as_tool(), summarizer.as_tool()],
    )

    brief = await coordinator.run_text("Give me a brief on the Voyager 1 probe.")
    print(brief)
    return brief


if __name__ == "__main__":
    asyncio.run(main())

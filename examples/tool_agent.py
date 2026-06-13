"""A tool-using agent on a live model.

Register a plain Python function as a tool, hand it to an `Agent`, and let the
model call it. Aether generates the tool's JSON Schema from the signature and
docstring, runs the LLM<->tool loop, and returns the final answer.

    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/tool_agent.py
"""
import asyncio

from aether import Agent, register_tool


@register_tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    return a + b


async def main() -> str:
    agent = Agent(
        "calculator",
        instructions="Use the available tools for any arithmetic.",
        tools=["add"],
    )
    answer = await agent.run_text("What is 17 + 25?")
    print(answer)
    return answer


if __name__ == "__main__":
    asyncio.run(main())

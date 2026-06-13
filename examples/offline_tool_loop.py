"""Tool-calling loop end to end — runs with NO API key.

A scripted fake model stands in for a real provider: first it asks to call the
`add` tool, then — given the result — it produces the final answer. Aether runs
the loop (dispatch the tool, feed the result back, ask again) and returns the
text. For the same thing against a live model, see `tool_agent.py`.

    python examples/offline_tool_loop.py
"""
import asyncio

from aether import Aether, register_tool
from aether.extensions.llm.fake import FakeProvider
from aether.llm.contracts import LLMResponse, ToolCall


@register_tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    return a + b


# A real provider decides to call the tool on its own; here we script the two
# turns the model would produce so the example needs no API key.
_fake = FakeProvider(responses=[
    LLMResponse(
        text="", model="fake-model", input_tokens=0, output_tokens=0,
        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 17, "b": 25})],
    ),
    LLMResponse(text="17 + 25 = 42", model="fake-model",
                input_tokens=0, output_tokens=0),
])


async def main() -> str:
    client = Aether(_fake)
    answer = await client.ask("What is 17 + 25?", tools=["add"])
    print(answer)
    return answer


if __name__ == "__main__":
    asyncio.run(main())

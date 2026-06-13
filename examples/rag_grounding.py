"""RAG with a grounding guarantee.

`GroundingGuard` is response middleware: it checks the model's answer against a
set of source documents and replaces anything they don't support with a
refusal. Swap `SOURCES` for the chunks your retriever returned. For production,
pass your own `verifier=` (e.g. an LLM judge) instead of the default heuristic.

    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/rag_grounding.py
"""
import asyncio

from aether import Aether, GroundingGuard

# Pretend these came back from a vector search for the user's question.
SOURCES = [
    "Aether is a Python framework for building LLM applications.",
    "Aether supports the OpenAI, Gemini, and Anthropic providers.",
]


async def main() -> None:
    client = Aether(middleware=[GroundingGuard(SOURCES)])

    # Supported by the documents -> answered normally.
    print("Q1:", await client.ask("Which providers does Aether support?"))

    # Not supported by the documents -> refused, not fabricated.
    print("Q2:", await client.ask("What is the capital of France?"))


if __name__ == "__main__":
    asyncio.run(main())

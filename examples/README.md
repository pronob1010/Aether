# Examples

Runnable examples showing what you can build with Aether.

## Run with no API key

`offline_tool_loop.py` uses a scripted fake model, so it runs immediately and
shows the tool-calling loop end to end:

```bash
python examples/offline_tool_loop.py
```

## Run against a real model

The rest call a live provider. Set one up first:

```bash
export LLM_PROVIDER=openai          # or: gemini, anthropic
export OPENAI_API_KEY=sk-...        # the matching *_API_KEY
```

| File | What it shows |
|------|---------------|
| `tool_agent.py` | An `Agent` that calls a Python function you registered as a tool. |
| `rag_grounding.py` | `GroundingGuard` — refuse answers the source documents don't support (RAG). |
| `multi_agent.py` | A coordinator that delegates sub-tasks to specialist sub-agents. |

```bash
python examples/tool_agent.py
python examples/rag_grounding.py
python examples/multi_agent.py
```

Every mechanic here is also covered offline (no keys) by the test suite —
`pytest` — if you want to see the behavior asserted rather than printed.

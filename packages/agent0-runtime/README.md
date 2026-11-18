# agent0-runtime

A small, headless agent runtime. It provides the model/tool loop, OpenAI-compatible
Responses backends, tools, sessions, structured events, cancellation, lifecycle hooks,
and plugins. Terminal interfaces and application policy belong to consumers such as
agent7.

## Minimal agent

Set `DEEPSEEK_API_KEY` or `KIMI_API_KEY`, then run:

```python
import asyncio
from agent0_runtime import Agent, tool

@tool("add")
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)

async def main():
    async with Agent.from_env(tools=["add"]) as agent:
        result = await agent.run("Use add to calculate 20 + 22.")
        print(result.text)

asyncio.run(main())
```

`Agent.from_env()` creates and owns the provider client, backend, and matching history
policy. Exiting the async context closes plugins and network resources. Pass an explicit
`provider/model` when more than one provider is configured.

Hosts with an existing client can share the same backend/history assembly without
transferring ownership:

```python
agent = Agent.from_openai(client, model, provider_id=provider_id, tools=registry)
```

This host-oriented factory defaults to `close_backend=False`.

Tools may also be registered as function objects or through `Tool.define()` when they
need an explicit schema or `ToolContext`. Reuse `RuntimeSession` for multi-turn history;
applications can use `MemorySessionStore` or implement `SessionStore` without
changing the agent loop. `ManagedProvider` and `FileCredentialStore` provide
host-configurable login, model discovery, and client cleanup.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

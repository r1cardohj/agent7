# Headless runtime and plugin API

`agent0_runtime` is the standalone headless runtime consumed by agent7. The old
`agent7.core` import path is temporarily preserved as a compatibility alias. The
runtime has no imports from the terminal, agent7 providers, or application-owned
session storage. It contains the generic loop plus optional built-in
filesystem/shell/web tools that other agent types can select without importing
agent7's coding profile.

## Quick start

The recommended construction path is `Agent.from_env()`. It discovers a configured
DeepSeek or Kimi credential, selects that provider's default model, and installs
the matching Responses backend and history policy:

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

Set `DEEPSEEK_API_KEY` or `KIMI_API_KEY` before running. With both configured,
pass an explicit reference such as `Agent.from_env("kimi/kimi-for-coding")`.
`AGENT0_MODEL` supplies the same default reference, and `AGENT0_BASE_URL` overrides
the endpoint. No tools are enabled unless supplied. The executable version of this
example is `examples/minimal_agent.py`.

A run without a session is independent. Reuse a `RuntimeSession` for conversation
history, as shown below. Entering the async context and exiting it closes plugins,
the backend, and its network client; no separate client cleanup is required.

## Typed runs and structured events

`run_agent_result()` remains the compatibility facade for agent7 hosts. Ordinary
embedders should prefer `Agent.from_env()`; the facade is useful when an existing
application already owns its client, routing, and coding registry:

```python
from agent7.agent import run_agent_result

result = await run_agent_result(
    "fix the tests",
    cwd=workspace,
    client=client,
    model="deepseek-flash",
    provider_id="deepseek",
    messages=messages,
    on_event=handle_event,
)

if result.stop_reason == "max_turns":
    show_blocked(result.turns)
else:
    show_answer(result.text)
```

`RunResult` contains `status`, `text`, `turns`, `run_id`, `finish_reason`,
`stop_reason`, and the latest token usage. Interruption and backend failures are
still exceptions, with corresponding terminal events emitted first.

Each `RuntimeEvent` has a `kind`, per-run sequence number, `run_id`, timestamp,
optional model-turn number, and a transport-safe payload. Event kinds cover
model/text/reasoning deltas, routing, usage, compaction, tool start/preview/end,
and terminal turn states. `event.to_dict()` is suitable for a future JSON-RPC or
WebSocket adapter. `on_event` may be synchronous or asynchronous.

For custom backends, instantiate `agent0_runtime.Agent` directly. The required
backend protocol is only `complete(request, events) -> ModelTurn`; `model`,
`provider_id`, `context_limit`, and `prepare()` are optional metadata/policy.
An owned backend may implement `close()` or `aclose()` to release resources:

```python
class Backend:
    async def complete(self, request, events) -> ModelTurn:
        ...
```

The runtime owns the model integration layer and normally depends on the
`openai` package. `agent0_runtime.OpenAIResponsesBackend` implements one
OpenAI-compatible Responses endpoint, including streaming, function calls,
usage mapping, retry-safe requests, raw Responses history, and client cleanup.
The runtime owns provider metadata, configurable clients, credential storage,
model discovery, and `RoutedResponsesBackend` intent/failover. Hosts such as
agent7 inject their credential store, client identity, and provider selection;
other applications can use the default provider registry, their own router, or
a different backend entirely.

`Agent(backend)` owns and closes its backend by default. Pass
`close_backend=False` when a host shares that backend or its network client across
runs. `Agent.close()` and async-context exit are idempotent.

Hosts that already own an OpenAI-compatible client should use the same default
assembly as `from_env()` without transferring client ownership:

```python
agent = Agent.from_openai(
    client,
    model,
    provider_id=provider_id,
    context_limit=context_limit,
    tools=registry,
)
```

This factory installs `OpenAIResponsesBackend` and `ResponsesHistoryPolicy`.
It defaults to `close_backend=False`; the host remains responsible for closing
the injected client.

## Tool registry

`@tool` derives names, JSON schemas, required arguments, and descriptions from a
Python signature and docstring. It works bare as `@tool`, as `@tool("name")`,
and configured as `@tool(name="...", description="...")`; synchronous and
asynchronous functions are supported. Common scalar, list, dictionary, literal,
and optional annotations are supported. Defaults become optional tool arguments.
The decorated value remains the original Python function, so tests and normal
calls can use it unchanged.

`Agent.from_env()` may receive decorated tool names such as `tools=["add"]`. The
constructor reflects the calling scope and resolves uniquely named tools. No
process-global registry is created; an ambiguous or invisible name raises a
clear error. Function objects remain the best choice for dynamically generated
or shadowed tools. Use `Tool.define()` when a tool needs an explicit schema or
`ToolContext`:

```python
from agent7.agent import create_tool_registry, run_agent_result
from agent0_runtime import Tool

registry = create_tool_registry()  # agent7's six coding tools
registry.register(
    Tool.define(
        "echo",
        "Return text",
        {"text": {"type": "string"}},
        lambda context, args: args["text"],
        required=["text"],
    )
)

result = await run_agent_result(
    "try echo", cwd=workspace, client=client, model="model",
    messages=[], registry=registry,
)
```

Handlers receive a `ToolContext` and a decoded argument dictionary. They may be
synchronous or asynchronous and return `str` or `ToolResult`. Synchronous
handlers run in a worker thread, matching agent7's existing behavior.

## Generic built-in tools

The runtime includes provider-neutral `read`, `write`, `edit`, `bash`, `fetch`,
and `search` definitions. Optional behavior is injected rather than imported:

```python
from agent0_runtime import BuiltinToolsConfig, create_builtin_registry, Shell

registry = create_builtin_registry(
    BuiltinToolsConfig(
        shell=Shell("bash", ("/bin/bash", "-lc")),
        search_provider=my_search_provider,  # any SearchProvider
    )
)
agent = Agent(backend, tools=registry)
```

The runtime ships `PillowImageReader` and `DeepSeekSearchProvider`; hosts may inject
credential resolvers or alternative implementations. Filesystem/shell tools require
a workspace only when selected by the model, so an assistant with no workspace can
still use its own tools. agent7 supplies its existing login credentials to the
runtime search provider without implementing the search transport itself.

## Sessions and stores

The core models only portable runtime state:

```python
from agent0_runtime import MemorySessionStore, RuntimeSession

session = RuntimeSession(
    id="assistant-1",
    messages=[],
    metadata={"persona": "general assistant"},
)
result = await agent.run("hello", session=session)
```

`RuntimeSession` contains an ID, message history, and application metadata.
`SessionStore` is the portable persistence contract; `MemorySessionStore` supports
tests and ephemeral agents. Applications own their durable formats and retention
policies. In particular, agent7 owns its workspace-scoped JSON snapshots, resume
migration, XDG paths, and non-overwriting trash in `agent7.file_sessions`.

For explicit per-run state, use `RunContext` and `Agent.run_in_context()`. The
legacy `cwd` parameter is the filesystem spelling of the optional run workspace;
neither is required for a non-filesystem agent.

## Cancellation

A host can cancel its asyncio task for immediate network-stream cancellation,
or pass a `CancellationToken` for cooperative cancellation:

```python
from agent0_runtime import CancellationToken

token = CancellationToken()
result_task = asyncio.create_task(agent.run(..., cancellation=token))
token.cancel("window closed")
await result_task  # raises AgentInterrupted
```

Tokens are checked between runtime boundaries and tools. A token cannot kill an
arbitrary synchronous plugin already running in a worker thread. On interruption,
all unanswered model tool calls receive an `error: interrupted` tool message so
history remains replayable.

## Lifecycle hooks

`LifecycleHooks` provides ordered, synchronous, read-only observers for session
and turn boundaries. `subscribe()` returns an unsubscribe callback. Observer
failures are isolated and recorded in `hooks.errors`; an optional `on_error`
callback can forward them to a host logger.

Lifecycle events distinguish the real `session_id` from `workspace_id`, and turn
events include the matching `run_id`. Terminal signal handling remains in the
REPL; outcome classification lives in `core.lifecycle`.

## Plugins

A plugin registers tools, backend factories, or observers during setup:

```python
class MyPlugin:
    api_version = 1

    def register(self, context):
        context.tools.register(make_my_tool())
        context.events.subscribe(log_event, kinds=["tool_finished"])

    async def close(self):  # optional
        ...
```

Load trusted objects directly with `Agent(..., plugins=[MyPlugin()])` or
`run_agent_result(..., plugins=[...])`.

Installed distributions can advertise an entry point:

```toml
[project.entry-points."agent0.plugins"]
my_plugin = "my_package:MyPlugin"
```

Discovery only reads metadata. Code is imported solely for names explicitly
passed as `plugin_names`:

```python
result = await run_agent_result(..., plugin_names=["my_plugin"])
```

There is deliberately no implicit load-all behavior. Python plugins have the
current user's full privileges and are not sandboxed. The first API supports
registration and orderly cleanup, not hot reload, arbitrary mutation of the
loop, or UI-specific hooks.

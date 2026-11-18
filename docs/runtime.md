# Headless runtime and plugin API

`agent7.core` is the internal boundary intended for a future standalone
`agent0-runtime` package. It has no imports from the terminal, agent7 providers,
or application-owned session storage. It contains the generic loop plus optional
built-in filesystem/shell/web tools that other agent types can select without
importing agent7's coding profile.

## Typed runs and structured events

The existing `run_agent()` remains a compatibility facade returning `str`.
New integrations should use `run_agent_result()`:

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

For custom backends, instantiate `agent7.core.Agent` directly. The required
backend protocol is only `complete(request, events) -> ModelTurn`; `model`,
`provider_id`, `context_limit`, and `prepare()` are optional metadata/policy:

```python
class Backend:
    async def complete(self, request, events) -> ModelTurn:
        ...
```

The runtime owns the model integration layer and normally depends on the
`openai` package. `agent7.core.OpenAIResponsesBackend` implements one
OpenAI-compatible Responses endpoint, including streaming, function calls,
usage mapping, retry-safe requests, and raw Responses history. The core also
has provider-neutral registry metadata for built-in OpenAI-compatible
providers such as DeepSeek and Kimi. agent7 adds only its auth-file/login UI
and `RoutedResponsesBackend` intent/failover policy; another application can
use the core backend/provider registry, its own router, or a different backend
entirely.

## Tool registry

Registries belong to an `Agent` instance; adding a plugin does not mutate global
handlers or another session:

```python
from agent7.agent import create_tool_registry, run_agent_result
from agent7.core import Tool

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
from agent7.core import BuiltinToolsConfig, create_builtin_registry, Shell

registry = create_builtin_registry(
    BuiltinToolsConfig(
        shell=Shell("bash", ("/bin/bash", "-lc")),
        search_provider=my_search_provider,  # any SearchProvider
    )
)
agent = Agent(backend, tools=registry)
```

Image decoding is optional through `ImageReader`; search credentials stay in the
application's `SearchProvider`. Filesystem/shell tools require a workspace only
when selected by the model, so an assistant with no workspace can still use its
own tools. agent7 injects its Pillow image reader and DeepSeek Search provider
from `agent7.tool_adapters`; neither is part of the generic tool definitions.

## Sessions and stores

The core models only portable runtime state:

```python
from agent7.core import MemorySessionStore, RuntimeSession

session = RuntimeSession(
    id="assistant-1",
    messages=[],
    metadata={"persona": "general assistant"},
)
result = await agent.run("hello", session=session)
```

`RuntimeSession` contains an ID, message history, and application metadata. It
does not prescribe names, credentials, paths, retention, workspace semantics, or
a storage format. `SessionStore` is the persistence contract; the included
`MemorySessionStore` is intended for tests and ephemeral agents. agent7's XDG
JSON snapshots remain an agent7 implementation, not runtime policy.

For explicit per-run state, use `RunContext` and `Agent.run_in_context()`. The
legacy `cwd` parameter is the filesystem spelling of the optional run workspace;
neither is required for a non-filesystem agent.

## Cancellation

A host can cancel its asyncio task for immediate network-stream cancellation,
or pass a `CancellationToken` for cooperative cancellation:

```python
from agent7.core import CancellationToken

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

# Headless runtime and plugin API

`agent7.core` is the internal boundary intended for a future standalone
`agent0-runtime` package. It has no imports from the terminal, agent7 providers,
session storage, or concrete coding tools. The CLI and a future GUI are both
hosts of this runtime.

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

For custom backends, instantiate `agent7.core.Agent` directly and implement:

```python
async def complete(request, events) -> ModelTurn:
    ...
```

The backend owns its wire protocol. The generic loop owns model/tool rounds,
history policy, cancellation, and run outcomes. An optional `prepare()` method
supports route selection before the first model request.

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

## Cancellation

A host can cancel its asyncio task for immediate network-stream cancellation,
or pass a `CancellationToken` for cooperative cancellation:

```python
from agent7.core import CancellationToken

token = CancellationToken()
result_task = asyncio.create_task(run(..., cancellation=token))
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

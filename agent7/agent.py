"""Host-side agent assembly for the agent7 CLI and development harnesses."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agent0_runtime import (
    DEFAULT_MAX_TURNS,
    CancellationToken,
    EventListener,
    LifecycleHooks,
    ModelBackend,
    ResponsesHistoryPolicy,
    RoutedResponsesBackend,
    RunResult,
    ToolRegistry,
    create_builtin_registry,
)

from .config import tool_summary
from .routing import FailoverRoute
from .tool_adapters import agent7_builtin_tools

RETRY_SLEEPS = (0.5, 1.5, 3.0)
STREAM_CLOSE_TIMEOUT = 2.0


def create_tool_registry() -> ToolRegistry:
    """Install agent7's configured providers into the runtime's generic tools."""

    return create_builtin_registry(agent7_builtin_tools())


async def run_agent_result(
    task: str,
    *,
    cwd: Path | None = None,
    client: Any = None,
    model: str = "",
    backend: ModelBackend | None = None,
    workspace: Any = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    messages: list[dict[str, Any]] | None = None,
    on_delta=None,
    on_thinking=None,
    on_answer_to_thinking=None,
    on_tool=None,
    on_tool_start=None,
    on_preview=None,
    on_usage=None,
    context_limit: int | None = None,
    prompt_tokens: int | None = None,
    on_compact=None,
    on_finish=None,
    on_model_start=None,
    provider_id: str | None = None,
    route: FailoverRoute | None = None,
    on_route=None,
    registry: ToolRegistry | None = None,
    on_event: EventListener | None = None,
    cancellation: CancellationToken | None = None,
    plugins: Iterable[Any] = (),
    plugin_names: Iterable[str] = (),
    hooks: LifecycleHooks | None = None,
    session: Any = None,
    run_id: str = "",
    system: str | None = None,
) -> RunResult:
    """Run one task with agent7's tools while keeping host resource ownership."""

    from agent0_runtime import Agent as CoreAgent

    runtime_options = {
        "tools": registry if registry is not None else create_tool_registry(),
        "lifecycle": hooks or getattr(session, "hooks", None),
        "plugins": plugins,
        "plugin_names": plugin_names,
    }
    if backend is None and route is None:
        if client is None or not model:
            raise ValueError("client and model are required when backend is not provided")
        runtime = CoreAgent.from_openai(
            client,
            model,
            provider_id=provider_id,
            context_limit=context_limit,
            retry_sleeps=RETRY_SLEEPS,
            stream_close_timeout=STREAM_CLOSE_TIMEOUT,
            # The CLI/session owns its client and may reuse it across many runs.
            close_backend=False,
            **runtime_options,
        )
    else:
        if backend is None:
            if client is None or not model:
                raise ValueError("client and model are required when backend is not provided")
            backend = RoutedResponsesBackend(
                client,
                model,
                route=route,
                provider_id=provider_id,
                context_limit=context_limit,
                on_route=on_route,
                retry_sleeps=RETRY_SLEEPS,
                stream_close_timeout=STREAM_CLOSE_TIMEOUT,
            )
        runtime = CoreAgent(
            backend,
            history=ResponsesHistoryPolicy(),
            close_backend=False,
            **runtime_options,
        )

    async def host_event(event) -> None:
        kind, data, turn = event.kind, event.payload, event.turn or 0
        callback: Callable[..., Any] | None = None
        args: tuple[Any, ...] = ()
        if kind == "text_delta":
            callback, args = on_delta, (data["text"],)
        elif kind == "reasoning_delta":
            callback, args = on_thinking, (data["text"],)
        elif kind == "answer_to_thinking":
            callback = on_answer_to_thinking
        elif kind == "model_started":
            callback, args = on_model_start, (turn,)
        elif kind == "model_finished":
            callback, args = on_finish, (data.get("finish_reason"),)
        elif kind == "usage":
            callback, args = on_usage, (data["usage"],)
        elif kind == "compacted":
            callback, args = on_compact, (data["changed"],)
        elif kind == "tool_started":
            callback, args = on_tool_start, (turn, data["name"], data["args"])
        elif kind == "tool_preview":
            callback, args = on_preview, (data["text"],)
        elif kind == "tool_finished" and on_tool is not None:
            callback = on_tool
            args = (turn, data["name"], data["args"], data["result"])

        if callback is None:
            if kind == "tool_finished":
                print(
                    f"  [{turn}] {data['name']} {tool_summary(data['name'], data['args'])}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        result = callback(*args)
        if result is not None and hasattr(result, "__await__"):
            await result

    unsubscribe = runtime.events.subscribe(host_event)
    try:
        return await runtime.run(
            task,
            cwd=cwd,
            messages=messages,
            workspace=workspace,
            max_turns=max_turns,
            context_limit=context_limit,
            prompt_tokens=prompt_tokens,
            cancellation=cancellation,
            on_event=on_event,
            session=session,
            run_id=run_id,
            system=system,
        )
    finally:
        unsubscribe()
        await runtime.close()

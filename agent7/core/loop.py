"""Headless model/tool agent loop with structured events and typed outcomes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .backend import BackendRegistry, ModelBackend, ModelRequest
from .events import EventBus, EventEmitter, EventListener
from .lifecycle import LifecycleHooks, run_with_lifecycle
from .plugins import PluginContext, PluginManager
from .tools import ToolContext, ToolExecutor, ToolRegistry
from .types import (
    AgentInterrupted,
    CancellationToken,
    Message,
    Recovery,
    RunResult,
    RunStatus,
    ToolCall,
    new_run_id,
)

DEFAULT_MAX_TURNS = 200


class HistoryPolicy(Protocol):
    """Provider/history policy used by the loop without knowing its wire format."""

    def estimate(self, messages: list[Message]) -> int: ...

    def should_compact(self, prompt_tokens: int | None, context_limit: int | None) -> bool: ...

    def compact(
        self, messages: list[Message], *, aggressive: bool = False
    ) -> tuple[list[Message], int]: ...

    def recover(
        self, messages: list[Message], error: Exception, *, attempt: int
    ) -> Recovery | None: ...


def _estimate(messages: list[Message]) -> int:
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    chars += sum(
        len(str((call.get("function") or {}).get("arguments") or ""))
        for message in messages
        for call in message.get("tool_calls") or []
    )
    return chars // 4 + 1


def _last_user_question(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            return content
    return None


def _answer_pending(messages: list[Message], calls: list[ToolCall]) -> None:
    """Keep every interrupted assistant tool batch replayable."""

    answered = {
        message.get("tool_call_id") for message in messages if message.get("role") == "tool"
    }
    for call in calls:
        if call.id and call.id not in answered:
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": "error: interrupted"}
            )


def _private_state(message: Message, state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key.startswith("_"):
            message[key] = value


class Agent:
    """Reusable headless runtime; one instance may serve multiple sequential runs.

    Backend/session implementations decide whether they support concurrent runs.
    Event subscription and tool/plugin registration are instance-scoped, while
    run IDs, event sequences, cancellation tokens, and outcomes are per run.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        tools: ToolRegistry | None = None,
        history: HistoryPolicy | None = None,
        events: EventBus | None = None,
        lifecycle: LifecycleHooks | None = None,
        backends: BackendRegistry | None = None,
        services: Mapping[str, Any] | None = None,
        plugins: Iterable[Any] = (),
        plugin_names: Iterable[str] = (),
    ) -> None:
        self.backend = backend
        self.tools = tools if tools is not None else ToolRegistry()
        self.history = history
        self.events = events if events is not None else EventBus()
        self.lifecycle = lifecycle if lifecycle is not None else LifecycleHooks()
        self.backends = backends if backends is not None else BackendRegistry()
        self.services = dict(services or {})
        self.plugins = PluginManager(
            PluginContext(
                tools=self.tools,
                backends=self.backends,
                events=self.events,
                lifecycle=self.lifecycle,
                services=self.services,
            )
        )
        for plugin in plugins:
            self.plugins.load(plugin)
        names = tuple(plugin_names)
        if names:
            self.plugins.load_discovered(names)

    async def close(self) -> None:
        await self.plugins.close()

    async def run(
        self,
        task: str,
        *,
        cwd: Path,
        messages: list[Message],
        max_turns: int = DEFAULT_MAX_TURNS,
        context_limit: int | None = None,
        prompt_tokens: int | None = None,
        cancellation: CancellationToken | None = None,
        on_event: EventListener | None = None,
        session: Any | None = None,
        run_id: str = "",
    ) -> RunResult:
        """Run one task, optionally wrapping it in Session lifecycle hooks."""

        run_id = run_id or new_run_id()
        listeners = (on_event,) if on_event is not None else ()
        events = self.events.for_run(run_id, listeners=listeners)
        work = self._run(
            task,
            cwd=cwd.resolve(),
            messages=messages,
            max_turns=max_turns,
            context_limit=context_limit,
            prompt_tokens=prompt_tokens,
            cancellation=cancellation,
            events=events,
        )
        if session is not None:
            return await run_with_lifecycle(session, work, run_id=run_id, hooks=self.lifecycle)
        return await work

    async def _run(
        self,
        task: str,
        *,
        cwd: Path,
        messages: list[Message],
        max_turns: int,
        context_limit: int | None,
        prompt_tokens: int | None,
        cancellation: CancellationToken | None,
        events: EventEmitter,
    ) -> RunResult:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        messages.append({"role": "user", "content": task})
        last_prompt = prompt_tokens
        last_usage: dict[str, int] | None = None
        finish_reason: str | None = None
        turns = recovery_attempts = 0
        pending: list[ToolCall] = []
        await events.emit("turn_started", task=task)
        try:
            if cancellation is not None:
                cancellation.throw_if_cancelled(run_id=events.run_id)
            prepare = getattr(self.backend, "prepare", None)
            if prepare is not None:
                estimate = self.history.estimate(messages) if self.history else _estimate(messages)
                await prepare(task, messages, estimate, events)
            while turns < max_turns:
                prompt = (
                    last_prompt
                    if last_prompt is not None
                    else (self.history.estimate(messages) if self.history else _estimate(messages))
                )
                backend_limit = self.backend.context_limit
                active_limit = backend_limit if backend_limit is not None else context_limit
                if self.history and self.history.should_compact(prompt, active_limit):
                    new_messages, changed = self.history.compact(messages)
                    if changed:
                        messages[:] = new_messages
                        last_prompt = None
                        await events.emit("compacted", changed=changed, aggressive=False)
                await events.emit("model_started", turn=turns + 1)
                try:
                    turn = await self.backend.complete(
                        ModelRequest(
                            messages=messages,
                            tools=self.tools.schemas(),
                            run_id=events.run_id,
                            turn=turns + 1,
                            cancellation=cancellation,
                        ),
                        events,
                    )
                except AgentInterrupted:
                    raise
                except Exception as error:
                    recovery_attempts += 1
                    recovery = (
                        self.history.recover(messages, error, attempt=recovery_attempts)
                        if self.history
                        else None
                    )
                    if recovery is None:
                        raise
                    messages[:] = recovery.messages
                    last_prompt = None
                    if recovery.compacted:
                        await events.emit("compacted", changed=recovery.compacted, aggressive=True)
                    continue

                recovery_attempts = 0
                turns += 1
                if turn.usage:
                    last_usage = turn.usage
                    last_prompt = turn.usage.get("prompt_tokens", last_prompt)
                    await events.emit("usage", turn=turns, usage=turn.usage)
                finish_reason = turn.finish_reason
                assistant: Message = {"role": "assistant", "content": turn.text}
                _private_state(assistant, turn.state)
                if turn.calls:
                    assistant["tool_calls"] = [call.as_message() for call in turn.calls]
                messages.append(assistant)
                if not turn.calls:
                    result = RunResult(
                        text=turn.text,
                        status=RunStatus.SUCCEEDED,
                        turns=turns,
                        run_id=events.run_id,
                        finish_reason=finish_reason,
                        usage=last_usage,
                    )
                    await events.emit(
                        "turn_succeeded",
                        turn=turns,
                        text=turn.text,
                        finish_reason=finish_reason,
                    )
                    return result

                pending = list(turn.calls)
                question = _last_user_question(messages)
                for call in pending:
                    if cancellation is not None:
                        cancellation.throw_if_cancelled(run_id=events.run_id)
                    execution = await self.executor.execute(
                        call,
                        ToolContext(
                            cwd=cwd,
                            call=call,
                            model=turn.model or self.backend.model,
                            user_question=question,
                            cancellation=cancellation,
                            services=self.services,
                        ),
                        events,
                        turn=turns,
                    )
                    tool_message: Message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": execution.result.text,
                    }
                    _private_state(tool_message, execution.result.state)
                    messages.append(tool_message)
                pending = []

            result = RunResult(
                text="",
                status=RunStatus.BLOCKED,
                turns=turns,
                run_id=events.run_id,
                finish_reason=finish_reason,
                stop_reason="max_turns",
                usage=last_usage,
            )
            await events.emit(
                "turn_blocked", turn=turns, reason="max_turns", text=result.legacy_text
            )
            return result
        except AgentInterrupted as error:
            _answer_pending(messages, pending)
            if error.partial:
                messages.append({"role": "assistant", "content": error.partial})
            error.run_id = error.run_id or events.run_id
            await events.emit(
                "turn_interrupted", turn=turns, reason=error.reason, partial=error.partial
            )
            raise
        except (asyncio.CancelledError, KeyboardInterrupt) as error:
            _answer_pending(messages, pending)
            interrupted = AgentInterrupted(run_id=events.run_id)
            await events.emit("turn_interrupted", turn=turns, reason="interrupted")
            raise interrupted from error
        except Exception as error:
            await events.emit(
                "turn_failed", turn=turns, error_type=type(error).__name__, reason=str(error)
            )
            raise

    @property
    def executor(self) -> ToolExecutor:
        # Constructing is cheap and keeps the public registry replaceable by an
        # embedder before a run, without a second stale registry reference.
        return ToolExecutor(self.tools)

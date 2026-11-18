"""Headless model/tool agent loop with structured events and typed outcomes."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .backend import BackendRegistry, ModelBackend, ModelRequest
from .events import EventBus, EventEmitter, EventListener
from .lifecycle import LifecycleHooks, run_with_lifecycle
from .plugins import PluginContext, PluginManager
from .session import RunContext, RuntimeSession
from .tools import (
    _TOOL_ATTRIBUTE,
    Tool,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    _as_tool,
)
from .types import (
    TOOL_ATTACHMENTS_KEY,
    AgentInterrupted,
    CancellationToken,
    Message,
    Recovery,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
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


def _merge_attachments(message: Message, result: ToolResult) -> None:
    if result.attachments:
        message[TOOL_ATTACHMENTS_KEY] = [block.to_dict() for block in result.attachments]


def _visible_tool_candidates(frame: Any | None) -> list[Tool]:
    """Collect uniquely identified decorated tools visible from the caller stack."""

    candidates: list[Tool] = []
    seen: set[int] = set()
    while frame is not None:
        for scope in (frame.f_locals, frame.f_globals):
            for value in scope.values():
                if id(value) in seen:
                    continue
                definition = getattr(value, _TOOL_ATTRIBUTE, None)
                tool = definition if isinstance(definition, Tool) else None
                if tool is None and isinstance(value, Tool):
                    tool = value
                if tool is not None:
                    seen.add(id(value))
                    candidates.append(tool)
        frame = frame.f_back
    return candidates


def _resolve_tools(
    values: Iterable[Tool | str | Callable[..., Any]],
    candidates: list[Tool],
) -> list[Tool]:
    """Resolve Tool objects, decorated callables, and decorated names."""

    resolved: list[Tool] = []
    for value in values:
        if not isinstance(value, str):
            resolved.append(_as_tool(value))
            continue
        matches: list[Tool] = []
        for candidate in candidates:
            if candidate.name == value and all(candidate is not seen for seen in matches):
                matches.append(candidate)
        if len(matches) != 1:
            qualifier = "not found" if not matches else "ambiguous"
            raise ValueError(f"tool name {value!r} is {qualifier}; pass the function instead")
        resolved.append(matches[0])
    return resolved


class Agent:
    """Reusable headless runtime; one instance may serve multiple sequential runs.

    Backend/session implementations decide whether they support concurrent runs.
    Event subscription and tool/plugin registration are instance-scoped, while
    run IDs, event sequences, cancellation tokens, and outcomes are per run.
    """

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        *,
        tools: ToolRegistry | Iterable[Tool | str | Callable[..., Any]] = (),
        context_limit: int | None = None,
        **runtime_options: Any,
    ) -> Agent:
        """Build the recommended default Agent from provider environment variables.

        With several configured providers, use an explicit ``provider/model``
        reference. The returned Agent owns and closes its OpenAI network client.
        Advanced runtime options are forwarded to the regular constructor.
        """

        from openai import AsyncOpenAI

        from .history import ResponsesHistoryPolicy
        from .openai_backend import OpenAIResponsesBackend
        from .providers import get_provider, resolve_all_env

        registry = tools if isinstance(tools, ToolRegistry) else None
        if registry is None:
            values = list(tools)
            if any(isinstance(value, str) for value in values):
                caller = inspect.currentframe().f_back
                try:
                    candidates = _visible_tool_candidates(caller)
                    registry = ToolRegistry(_resolve_tools(values, candidates))
                finally:
                    # Never retain caller frame references on this Agent.
                    del caller
            else:
                registry = ToolRegistry(values)
        requested = (model or os.environ.get("AGENT7_MODEL", "")).strip()
        credentials = resolve_all_env()
        if not credentials:
            raise ValueError("no model provider configured; set DEEPSEEK_API_KEY or KIMI_API_KEY")
        available = [provider for key in credentials if (provider := get_provider(key))]
        if requested == "auto":
            raise ValueError("Agent.from_env() needs a concrete provider/model, not 'auto'")
        if "/" in requested:
            provider_name, model_name = requested.split("/", 1)
            provider = get_provider(provider_name)
            if provider is None or not model_name:
                raise ValueError(f"invalid model reference: {requested!r}")
        elif requested:
            if len(available) != 1:
                raise ValueError("use provider/model when more than one provider is configured")
            provider, model_name = available[0], requested
        else:
            provider = available[0]
            model_name = provider.default_model
        credential = credentials.get(provider.id)
        if credential is None:
            raise ValueError(f"{requested!r} requires {provider.env_keys[0]}")
        client = AsyncOpenAI(
            api_key=credential.api_key,
            base_url=os.environ.get("AGENT7_BASE_URL", "").strip() or credential.base_url,
        )
        backend = OpenAIResponsesBackend(
            client,
            model_name,
            provider_id=provider.id,
            context_limit=context_limit,
        )
        runtime_options.setdefault("history", ResponsesHistoryPolicy())
        runtime_options.setdefault("close_backend", True)
        return cls(backend, tools=registry, **runtime_options)

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
        close_backend: bool = True,
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
        self.close_backend = close_backend
        self._closed = False
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

    async def __aenter__(self) -> Agent:
        if self._closed:
            raise RuntimeError("agent is closed")
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close plugins and, when owned, the backend and its network resources."""

        if self._closed:
            return
        self._closed = True
        try:
            await self.plugins.close()
        finally:
            if self.close_backend:
                close = getattr(self.backend, "aclose", None) or getattr(
                    self.backend, "close", None
                )
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    async def run(
        self,
        task: str,
        *,
        cwd: Path | None = None,
        messages: list[Message] | None = None,
        workspace: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        context_limit: int | None = None,
        prompt_tokens: int | None = None,
        cancellation: CancellationToken | None = None,
        on_event: EventListener | None = None,
        session: Any | None = None,
        run_id: str = "",
    ) -> RunResult:
        """Run one task in a session, optionally wrapping it in lifecycle hooks.

        ``cwd`` is the legacy filesystem spelling of ``workspace``. Non-filesystem
        agents may omit both; filesystem/shell tools then report a clear error if
        a model nevertheless selects them.
        """

        runtime_session = session if isinstance(session, RuntimeSession) else None
        if runtime_session is not None and messages is None:
            messages = runtime_session.messages
        elif messages is None:
            # A one-shot run needs no explicit session or mutable message list.
            messages = []
        selected_workspace = workspace if workspace is not None else cwd
        if selected_workspace is None and runtime_session is not None:
            metadata_workspace = runtime_session.metadata.get("workspace")
            selected_workspace = metadata_workspace
        if runtime_session is None:
            session_id = (
                getattr(session, "id", "") or getattr(session, "session_id", "") or "session"
            )
            runtime_session = RuntimeSession(
                id=str(session_id),
                messages=messages,
                metadata={},
            )
        context = RunContext(
            session=runtime_session,
            run_id=run_id or new_run_id(),
            workspace=selected_workspace,
            cancellation=cancellation,
        )
        work = self.run_in_context(
            context,
            task,
            max_turns=max_turns,
            context_limit=context_limit,
            prompt_tokens=prompt_tokens,
            on_event=on_event,
        )
        if session is not None:
            return await run_with_lifecycle(
                session, work, run_id=context.run_id, hooks=self.lifecycle
            )
        return await work

    async def run_in_context(
        self,
        context: RunContext,
        task: str,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        context_limit: int | None = None,
        prompt_tokens: int | None = None,
        on_event: EventListener | None = None,
    ) -> RunResult:
        """Run one task using an explicit per-run context."""

        if self._closed:
            raise RuntimeError("agent is closed")
        listeners = (on_event,) if on_event is not None else ()
        events = self.events.for_run(context.run_id, listeners=listeners)
        services = {**self.services, **dict(context.services)}
        return await self._run(
            task,
            workspace=context.workspace,
            messages=context.session.messages,
            session_id=context.session.id,
            max_turns=max_turns,
            context_limit=context_limit,
            prompt_tokens=prompt_tokens,
            cancellation=context.cancellation,
            services=services,
            events=events,
        )

    async def _run(
        self,
        task: str,
        *,
        workspace: Any | None,
        messages: list[Message],
        session_id: str,
        max_turns: int,
        context_limit: int | None,
        prompt_tokens: int | None,
        cancellation: CancellationToken | None,
        services: Mapping[str, Any],
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
                backend_limit = getattr(self.backend, "context_limit", None)
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
                        session_id=session_id,
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
                            cwd=workspace,
                            call=call,
                            model=turn.model or getattr(self.backend, "model", None),
                            user_question=question,
                            cancellation=cancellation,
                            services=services,
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
                    _merge_attachments(tool_message, execution.result)
                    messages.append(tool_message)
                pending = []

            result = RunResult(
                text="",
                status=RunStatus.BLOCKED,
                turns=turns,
                run_id=events.run_id,
                session_id=session_id,
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

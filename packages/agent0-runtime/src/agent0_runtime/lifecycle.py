"""Read-only session/turn lifecycle hooks and run outcome classification."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

from .types import AgentInterrupted, RunResult, RunStatus, new_run_id

LifecycleKind = Literal[
    "session_started",
    "turn_started",
    "turn_succeeded",
    "turn_interrupted",
    "turn_blocked",
    "turn_failed",
    "session_ended",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_key(cwd: Path) -> str:
    return hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LifecycleContext:
    """Minimal session identity for lifecycle observers."""

    session_id: str
    cwd: Path | None = None
    model: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LifecycleEvent:
    kind: LifecycleKind
    seq: int
    session_id: str
    cwd: Path | None = None
    model: str = ""
    reason: str = ""
    run_id: str = ""
    workspace_id: str = ""
    timestamp: str = field(default_factory=_timestamp)


LifecycleListener: TypeAlias = Callable[[LifecycleEvent], Any]


@dataclass(frozen=True)
class LifecycleListenerError:
    error: Exception
    event: LifecycleEvent
    listener: LifecycleListener


LifecycleErrorListener = Callable[[LifecycleListenerError], Any]


class LifecycleHooks:
    """Ordered synchronous observers with fault isolation and unsubscribe.

    Observers should be non-blocking. Async application integrations belong on
    ``EventBus``; lifecycle hooks intentionally retain a tiny synchronous API.
    """

    def __init__(self, *, on_error: LifecycleErrorListener | None = None) -> None:
        self._listeners: list[LifecycleListener] = []
        self._seq = 0
        self._on_error = on_error
        self.errors: list[LifecycleListenerError] = []

    def subscribe(self, listener: LifecycleListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def emit(
        self,
        kind: LifecycleKind,
        session: Any,
        *,
        reason: str = "",
        run_id: str = "",
    ) -> None:
        self._seq += 1
        if isinstance(session, LifecycleContext):
            context = session
        else:
            raw_cwd = getattr(session, "cwd", None) or getattr(session, "workspace", None)
            cwd = Path(raw_cwd).resolve() if raw_cwd is not None else None
            metadata = getattr(session, "metadata", None)
            session_id = (
                getattr(session, "id", "")
                or getattr(session, "session_id", "")
                or (metadata.get("session_id", "") if isinstance(metadata, Mapping) else "")
            )
            context = LifecycleContext(
                session_id=str(session_id or "session"),
                cwd=cwd,
                model=str(
                    getattr(session, "model_ref", "")
                    or getattr(session, "model", "")
                    or (metadata.get("model", "") if isinstance(metadata, Mapping) else "")
                ),
                metadata=metadata if isinstance(metadata, Mapping) else {},
            )
        workspace_id = _workspace_key(context.cwd) if context.cwd is not None else ""
        event = LifecycleEvent(
            kind=kind,
            seq=self._seq,
            session_id=context.session_id,
            cwd=context.cwd,
            model=context.model,
            reason=reason,
            run_id=run_id,
            workspace_id=workspace_id,
        )
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception as error:
                failure = LifecycleListenerError(error, event, listener)
                self.errors.append(failure)
                if self._on_error is not None:
                    with suppress(Exception):
                        self._on_error(failure)


def emit_lifecycle(
    session: Any,
    kind: LifecycleKind,
    *,
    reason: str = "",
    run_id: str = "",
) -> None:
    """Emit through a Session, tolerating Session-like embedder objects."""

    hooks = getattr(session, "hooks", None)
    if hooks is not None:
        hooks.emit(kind, session, reason=reason, run_id=run_id)


async def run_with_lifecycle(
    session: Any,
    work: Awaitable[RunResult | str],
    *,
    run_id: str = "",
    hooks: LifecycleHooks | None = None,
) -> RunResult:
    """Classify one runner using typed outcomes and emit exactly one end event.

    ``str`` is accepted temporarily for older agent7 embedders. New runners
    return ``RunResult`` and never need to encode control state in answer text.
    """

    run_id = run_id or new_run_id()
    hooks = hooks or getattr(session, "hooks", None)
    if hooks is not None:
        hooks.emit("turn_started", session, run_id=run_id)
    try:
        outcome = await work
    except (AgentInterrupted, asyncio.CancelledError, KeyboardInterrupt) as error:
        if hooks is not None:
            reason = error.reason if isinstance(error, AgentInterrupted) else "interrupted"
            hooks.emit("turn_interrupted", session, reason=reason, run_id=run_id)
        raise
    except Exception as error:
        if hooks is not None:
            hooks.emit("turn_failed", session, reason=type(error).__name__, run_id=run_id)
        raise
    result = (
        outcome if isinstance(outcome, RunResult) else RunResult.from_legacy(outcome, run_id=run_id)
    )
    if hooks is not None:
        if result.status is RunStatus.BLOCKED:
            hooks.emit(
                "turn_blocked",
                session,
                reason=result.stop_reason or "blocked",
                run_id=run_id,
            )
        else:
            hooks.emit("turn_succeeded", session, run_id=run_id)
    return result

"""Read-only lifecycle hooks for host integrations.

Hooks observe application boundaries; they cannot change the agent loop, messages,
or tools. A broken observer must never break the task it is observing.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .store import workspace_key

LifecycleKind = Literal[
    "session_started",
    "turn_started",
    "turn_succeeded",
    "turn_interrupted",
    "turn_blocked",
    "turn_failed",
    "session_ended",
]

@dataclass(frozen=True)
class LifecycleEvent:
    kind: LifecycleKind
    seq: int
    session_id: str
    cwd: Path
    model: str
    reason: str = ""

LifecycleListener = Callable[[LifecycleEvent], None]

class LifecycleHooks:
    """Ordered, synchronous observers with fault isolation."""

    def __init__(self) -> None:
        self._listeners: list[LifecycleListener] = []
        self._seq = 0

    def subscribe(self, listener: LifecycleListener) -> None:
        self._listeners.append(listener)

    def emit(self, kind: LifecycleKind, session: Any, *, reason: str = "") -> None:
        self._seq += 1
        cwd = session.cwd.resolve()
        event = LifecycleEvent(
            kind=kind,
            seq=self._seq,
            session_id=workspace_key(cwd),
            cwd=cwd,
            model=getattr(session, "model_ref", None) or session.model,
            reason=reason,
        )
        for listener in tuple(self._listeners):
            # Observability is optional; an integration cannot be allowed to
            # interrupt a model request or leave tool calls unanswered.
            with suppress(Exception):
                listener(event)

def emit_lifecycle(session: Any, kind: LifecycleKind, *, reason: str = "") -> None:
    """Emit through a Session, tolerating Session-like objects used by embedders."""
    hooks = getattr(session, "hooks", None)
    if hooks is not None:
        hooks.emit(kind, session, reason=reason)

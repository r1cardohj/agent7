"""Shared, transport-neutral values used by the agent runtime.

This module deliberately knows nothing about OpenAI, agent7 sessions, or a UI.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Message = dict[str, Any]
JSONState = dict[str, Any]
# Generic tool-result attachment blocks carried next to a text message.
TOOL_ATTACHMENTS_KEY = "_attachments"


def new_run_id() -> str:
    """Return an ID suitable for logs, event streams, and future RPC transport."""

    return f"run-{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation with its encoded JSON arguments."""

    id: str
    name: str
    arguments: str = "{}"

    def as_message(self) -> dict[str, Any]:
        """Project to the portable assistant/tool history shape."""

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass
class ModelTurn:
    """One completed model round, normalized across backend implementations.

    ``state`` may contain JSON-safe, private message fields (named ``_*``). A
    Responses adapter uses this for raw replay items without leaking that wire
    concept into the generic loop.
    """

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    model: str | None = None
    state: JSONState = field(default_factory=dict)


@dataclass(frozen=True)
class Attachment:
    """One model-facing attachment, independent of any provider wire format."""

    data: str
    mime_type: str
    kind: str = "image"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "mime_type": self.mime_type, "data": self.data}


@dataclass(frozen=True)
class ToolResult:
    """Text shown to the model plus provider-neutral result extensions."""

    text: str
    state: JSONState = field(default_factory=dict)
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class Recovery:
    """A history edit that makes one failed model request safe to retry."""

    messages: list[Message]
    compacted: int = 0


class RunStatus(str, Enum):
    """Terminal states for a normally completed run.

    Interruption and backend failure remain exceptions; their corresponding
    lifecycle events are emitted before those exceptions cross the API boundary.
    """

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RunResult:
    """The typed outcome of an agent run."""

    text: str
    status: RunStatus
    turns: int
    run_id: str
    session_id: str = ""
    finish_reason: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] | None = None

    @property
    def legacy_text(self) -> str:
        """The pre-runtime string result kept for agent7 API compatibility."""

        if self.status is RunStatus.BLOCKED and self.stop_reason == "max_turns":
            return f"(stopped after {self.turns} turns)"
        return self.text

    @classmethod
    def from_legacy(cls, text: str, *, run_id: str = "") -> RunResult:
        """Adapt an older runner while hosts migrate to typed results."""

        match = re.fullmatch(r"\(stopped after (\d+) turns\)", text)
        if match:
            return cls(
                text="",
                status=RunStatus.BLOCKED,
                turns=int(match.group(1)),
                run_id=run_id,
                stop_reason="max_turns",
            )
        return cls(text=text, status=RunStatus.SUCCEEDED, turns=0, run_id=run_id)


class AgentInterrupted(Exception):
    """A cancelled run, optionally carrying text emitted before interruption."""

    def __init__(self, partial: str = "", *, reason: str = "interrupted", run_id: str = "") -> None:
        self.partial = partial
        self.reason = reason
        self.run_id = run_id
        super().__init__(partial)


class CancellationToken:
    """Thread-safe cooperative cancellation for hosts that do not own a Task.

    Cancelling the asyncio task remains the way to interrupt a blocked network
    stream immediately. A token is checked at runtime and backend boundaries;
    like ``asyncio.to_thread``, it cannot kill an arbitrary synchronous plugin.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "cancelled"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._reason = reason or "cancelled"
            self._event.set()

    def throw_if_cancelled(self, *, partial: str = "", run_id: str = "") -> None:
        if self.cancelled:
            raise AgentInterrupted(partial, reason=self.reason, run_id=run_id)

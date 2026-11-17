"""Provider-neutral session state and storage contracts.

A session is runtime state, not an application policy: applications own paths,
formats, names, credentials, retention, and picker/trash behavior.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import CancellationToken, Message


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex}"


@dataclass
class RuntimeSession:
    """The minimal state needed to continue one conversation."""

    id: str = field(default_factory=new_session_id)
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> RuntimeSession:
        """Snapshot this session without sharing mutable message/state objects."""

        return RuntimeSession(
            id=self.id,
            messages=copy.deepcopy(self.messages),
            metadata=copy.deepcopy(self.metadata),
        )


class SessionStore(Protocol):
    """Storage backend contract; implementations choose sync/async and format."""

    async def load(self, session_id: str) -> RuntimeSession: ...

    async def save(self, session: RuntimeSession) -> None: ...


class MemorySessionStore:
    """Small in-memory SessionStore for tests, examples, and ephemeral agents."""

    def __init__(self, sessions: Mapping[str, RuntimeSession] | None = None) -> None:
        self._sessions = dict(sessions or {})

    async def load(self, session_id: str) -> RuntimeSession:
        try:
            return self._sessions[session_id].clone()
        except KeyError:
            raise ValueError(f"unknown session: {session_id}") from None

    async def save(self, session: RuntimeSession) -> None:
        self._sessions[session.id] = session.clone()

    def ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)


@dataclass(frozen=True)
class RunContext:
    """Everything scoped to one invocation of a runtime session."""

    session: RuntimeSession
    run_id: str
    workspace: Any | None = None
    services: Mapping[str, Any] = field(default_factory=dict)
    cancellation: CancellationToken | None = None

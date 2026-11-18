"""Ordered structured events for terminal, GUI, and plugin observers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias

RuntimeEventKind = Literal[
    "turn_started",
    "model_started",
    "text_delta",
    "reasoning_delta",
    "answer_to_thinking",
    "model_finished",
    "route_changed",
    "usage",
    "compacted",
    "tool_started",
    "tool_preview",
    "tool_finished",
    "turn_succeeded",
    "turn_blocked",
    "turn_interrupted",
    "turn_failed",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuntimeEvent:
    """One serializable event in a single run's ordered stream."""

    kind: RuntimeEventKind
    seq: int
    run_id: str
    timestamp: str = field(default_factory=_timestamp)
    turn: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seq": self.seq,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "turn": self.turn,
            "payload": dict(self.payload),
        }


EventListener: TypeAlias = Callable[[RuntimeEvent], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ListenerError:
    """An isolated observer failure, available for logging and diagnostics."""

    error: Exception
    event: RuntimeEvent
    listener: EventListener


@dataclass(frozen=True)
class _Subscription:
    listener: EventListener
    kinds: frozenset[str] | None = None


ErrorListener = Callable[[ListenerError], Any]


class EventBus:
    """Process-local subscriptions; each run gets an independent sequence/ID.

    Observer failures are isolated from the agent loop and recorded in
    ``errors``. An optional error listener can forward them to a host logger.
    Event delivery is ordered and backpressured: an async GUI subscriber must
    finish handling one event before the next is delivered to that subscriber.
    """

    def __init__(self, *, on_error: ErrorListener | None = None) -> None:
        self._subscriptions: list[_Subscription] = []
        self._on_error = on_error
        self.errors: list[ListenerError] = []

    def subscribe(
        self, listener: EventListener, *, kinds: Iterable[RuntimeEventKind] | None = None
    ) -> Callable[[], None]:
        """Subscribe and return an idempotent unsubscribe callback."""

        subscription = _Subscription(listener, frozenset(kinds) if kinds is not None else None)
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._subscriptions.remove(subscription)

        return unsubscribe

    def for_run(self, run_id: str, *, listeners: Iterable[EventListener] = ()) -> EventEmitter:
        local = [_Subscription(listener) for listener in listeners]
        return EventEmitter(self, run_id, local)

    async def _dispatch(self, event: RuntimeEvent, local: list[_Subscription]) -> None:
        for subscription in (*self._subscriptions, *local):
            if subscription.kinds is not None and event.kind not in subscription.kinds:
                continue
            try:
                result = subscription.listener(event)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    await result
            except Exception as error:  # observers must not break a model/tool round
                failure = ListenerError(error, event, subscription.listener)
                self.errors.append(failure)
                if self._on_error is not None:
                    # Diagnostics are observers too; never recursively fail a run.
                    with suppress(Exception):
                        self._on_error(failure)


class EventEmitter:
    """The write side of one run's event stream, passed to backends/tools."""

    def __init__(
        self, bus: EventBus, run_id: str, local: list[_Subscription] | None = None
    ) -> None:
        self._bus = bus
        self.run_id = run_id
        self._local = local or []
        self._seq = 0

    async def emit(
        self, kind: RuntimeEventKind, *, turn: int | None = None, **payload: Any
    ) -> RuntimeEvent:
        self._seq += 1
        event = RuntimeEvent(
            kind=kind,
            seq=self._seq,
            run_id=self.run_id,
            turn=turn,
            payload=payload,
        )
        await self._bus._dispatch(event, self._local)
        return event

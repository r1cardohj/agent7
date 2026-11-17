"""Model backend SPI and registry.

A backend owns provider wire formats and streaming. The generic loop owns model
rounds, tool rounds, history, cancellation, and terminal run outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .types import CancellationToken, Message, ModelTurn

if TYPE_CHECKING:
    from .events import EventEmitter


@dataclass(frozen=True)
class ModelRequest:
    """Everything a backend needs for one model round."""

    messages: list[Message]
    tools: list[dict[str, Any]]
    run_id: str
    turn: int = 1
    cancellation: CancellationToken | None = None


@runtime_checkable
class ModelBackend(Protocol):
    """The required backend boundary; ``prepare`` is an optional extension."""

    @property
    def model(self) -> str: ...

    @property
    def provider_id(self) -> str | None: ...

    @property
    def context_limit(self) -> int | None: ...

    async def complete(self, request: ModelRequest, events: EventEmitter) -> ModelTurn: ...


BackendFactory = Callable[..., ModelBackend]


class BackendRegistry:
    """Named backend factories, primarily populated by plugins."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory, *, replace: bool = False) -> None:
        key = name.strip()
        if not key:
            raise ValueError("backend name cannot be empty")
        if not callable(factory):
            raise TypeError("backend factory must be callable")
        if key in self._factories and not replace:
            raise ValueError(f"backend already registered: {key}")
        self._factories[key] = factory

    def create(self, name: str, **kwargs: Any) -> ModelBackend:
        try:
            factory = self._factories[name]
        except KeyError:
            raise ValueError(f"unknown backend: {name}") from None
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

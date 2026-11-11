"""Logical model routes and safe, ordered failover targets.

`auto` is deliberately not a provider: it owns no credentials or base URL.  A
route is an ordered set of concrete provider/model targets.  Selection policies
(such as intent classification) can later choose or reorder these targets without
changing the agent loop's replay and failover rules.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from openai import AsyncOpenAI

from .providers import ApiKeyProvider, Credential, get_provider

AUTO_REF = "auto"
AUTO_ENV = "AGENT7_AUTO_MODELS"

@dataclass(frozen=True)
class RouteTarget:
    """One concrete upstream request destination."""

    provider_id: str
    model: str
    credential: Credential
    client: AsyncOpenAI
    context_limit: int | None = None

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model}"

@dataclass
class FailoverRoute:
    """Ordered targets; the current target is sticky until a safe failover."""

    targets: tuple[RouteTarget, ...]
    skipped: tuple[str, ...] = ()
    index: int = 0

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("auto: a route needs at least one target")

    @property
    def current(self) -> RouteTarget:
        return self.targets[self.index]

    def advance(self) -> RouteTarget | None:
        """Move to the next target, or return None after the last one."""
        if self.index + 1 >= len(self.targets):
            return None
        self.index += 1
        return self.current

    async def close(self) -> None:
        seen: set[int] = set()
        for target in self.targets:
            client: Any = target.client
            if id(client) in seen:
                continue
            seen.add(id(client))
            with suppress(Exception):
                result = client.close()
                if asyncio.iscoroutine(result):
                    await result

def auto_configured() -> bool:
    return bool(os.environ.get(AUTO_ENV, "").strip())

def parse_auto_refs(raw: str) -> list[tuple[str, str]]:
    """Parse comma-separated provider/model refs without guessing providers."""
    if not raw.strip():
        raise ValueError(f"auto: set {AUTO_ENV}=provider/model,provider/model")
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        provider, sep, model = value.partition("/")
        if not sep or not provider.strip() or not model.strip():
            raise ValueError(f"auto: invalid target {value!r}; expected provider/model")
        if provider.strip().lower() == AUTO_REF:
            raise ValueError("auto: nested auto routes are not supported")
        key = value.lower()
        if key not in seen:
            refs.append((provider.strip(), model.strip()))
            seen.add(key)
    if not refs:
        raise ValueError(f"auto: set {AUTO_ENV}=provider/model,provider/model")
    return refs

def build_auto_route(
    credentials: dict[str, Credential], *, raw: str | None = None,
    context_limit: Callable[[str], int | None] | None = None,
) -> FailoverRoute:
    """Resolve configured refs and create clients; logged-out targets are skipped."""
    specs = parse_auto_refs(os.environ.get(AUTO_ENV, "") if raw is None else raw)
    prepared: list[tuple[ApiKeyProvider, str, Credential]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for provider_name, model in specs:
        provider = get_provider(provider_name)
        if provider is None:
            raise ValueError(f"auto: unknown provider {provider_name!r}")
        ref = f"{provider.id}/{model}"
        if ref.lower() in seen:
            continue
        seen.add(ref.lower())
        credential = credentials.get(provider.id) or provider.resolve()
        if credential is None:
            skipped.append(ref)
            continue
        prepared.append((provider, model, credential))
    if not prepared:
        detail = f"; logged out: {', '.join(skipped)}" if skipped else ""
        raise ValueError(f"auto: no usable targets{detail}")
    targets = tuple(
        RouteTarget(
            provider_id=provider.id,
            model=model,
            credential=credential,
            client=provider.make_client(credential),
            context_limit=context_limit(model) if context_limit else None,
        )
        for provider, model, credential in prepared
    )
    return FailoverRoute(targets=targets, skipped=tuple(skipped))

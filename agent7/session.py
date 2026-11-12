"""REPL / oneshot session state."""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .agent import DEFAULT_MAX_TURNS
from .hooks import LifecycleHooks
from .providers import Credential, Provider, get_provider
from .routing import AUTO_REF, FailoverRoute, RouteTarget

async def _close_client(client: Any) -> None:
    if client is None:
        return
    with suppress(Exception):
        result = client.close()
        if asyncio.iscoroutine(result):
            await result

def resolve_context_limit(model: str) -> int | None:
    raw = os.environ.get("AGENT7_CONTEXT_LIMIT", "").strip()
    if raw:
        try:
            return int(raw.replace("_", "").replace(",", ""))
        except ValueError:
            pass
    key = model.lower().strip()
    if m := re.search(r"(?<![a-z0-9])(\d+)\s*k(?![a-z0-9])", key):
        return int(m.group(1)) * 1000
    if m := re.search(r"(?<![a-z0-9])(\d+)\s*m(?![a-z0-9])", key):
        return int(m.group(1)) * 1_000_000
    if "deepseek" in key:
        return 1_048_576 if "flash" in key else 128_000
    if "grok" in key:
        if "grok-4.7" in key:
            return 500_000
        return 256_000 if ("grok-4" in key or "build" in key) else 131_072
    if "kimi-for-coding" in key:
        return 262_144
    if key == "k3" or key.startswith("k3.1"):
        return 1_048_576
    return None

def fmt_tokens(n: int) -> str:
    if n < 10_000:
        return f"{n:,}"
    k = n / 1000
    return f"{k:.0f}k" if k >= 100 or abs(k - round(k)) < 0.05 else f"{k:.1f}k"

def fmt_limit(n: int) -> str:
    """A context limit in the unit its vendor prints: 1M, 256k, 128k.

    A model sold as 1M has 1,048,576 tokens, so the decimal formatter used for
    real token counts would print 1049k — the same window under a different
    base. Match the label instead: pick whichever base divides the number
    exactly, so 1,048,576 → 1M, 262,144 → 256k, 256,000 → 256k.
    """
    for suffix, base in (("M", 1_000_000), ("M", 1024 ** 2), ("k", 1_000), ("k", 1024)):
        if n % base == 0 and 1 <= n // base < 1000:
            return f"{n // base}{suffix}"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    return fmt_tokens(n)

def format_context_prompt(session: Session) -> str:
    u = fmt_tokens(session.prompt_tokens) if session.prompt_tokens is not None else "?"
    return u if session.context_limit is None else f"{u}/{fmt_limit(session.context_limit)}"

def context_pressure(session: Session) -> int | None:
    """Share of the context window in use, as a percentage, or None if unknown."""
    if session.prompt_tokens is None or not session.context_limit:
        return None
    return min(100, round(100 * session.prompt_tokens / session.context_limit))

def fmt_cwd(path: Path, *, limit: int = 44) -> str:
    """The workspace as `~`-relative text, shortened from the left when long."""
    text, home = str(path), str(Path.home())
    if text == home:
        text = "~"
    elif text.startswith(home.rstrip("/") + "/"):
        text = "~" + text[len(home):]
    if limit > 1 and len(text) > limit:
        tail = text[1 - limit:]
        cut = tail.find("/")
        text = "…" + (tail[cut:] if 0 <= cut < len(tail) - 1 else tail)
    return text

def format_usage(session: Session) -> str:
    if session.prompt_tokens is None:
        return "usage=none" + (
            f"  context_limit={fmt_limit(session.context_limit)}" if session.context_limit else ""
        )
    parts = [
        f"prompt_tokens={fmt_tokens(session.prompt_tokens)}",
        f"completion_tokens={fmt_tokens(session.completion_tokens or 0)}",
    ]
    if session.total_tokens is not None:
        parts.append(f"total_tokens={fmt_tokens(session.total_tokens)}")
    if (pct := context_pressure(session)) is not None:
        parts.append(f"context_limit={fmt_limit(session.context_limit)} ({pct}%)")
    return "  ".join(parts)

@dataclass
class Session:
    model: str
    cwd: Path
    client: AsyncOpenAI
    max_turns: int = DEFAULT_MAX_TURNS
    messages: list[dict[str, Any]] = field(default_factory=list)
    models_cache: list[str] | None = None
    model_ref: str | None = None
    context_limit: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider_id: str | None = None
    cred_source: str | None = None
    credential: Credential | None = None
    credentials: dict[str, Credential] = field(default_factory=dict)
    route: FailoverRoute | None = None
    active_model_ref: str | None = None
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks, repr=False)

    def set_model(self, model: str, *, ref: str | None = None) -> None:
        self.model = model
        self.model_ref = ref or (f"{self.provider_id}/{model}" if self.provider_id else model)
        self.active_model_ref = None
        self.context_limit = resolve_context_limit(model)

    @property
    def display_model(self) -> str:
        logical = self.model_ref or self.model
        if self.route is not None and self.active_model_ref:
            shown = f"{logical} → {self.active_model_ref}"
            selection = self.route.selection
            if selection is not None:
                shown += f" · {selection.intent}"
                if (score := selection.scores.get(self.active_model_ref)) is not None:
                    shown += f" · {score:.0f}"
            return shown
        return logical

    def update_usage(self, usage: dict[str, int]) -> None:
        self.prompt_tokens = usage.get("prompt_tokens")
        self.completion_tokens = usage.get("completion_tokens")
        self.total_tokens = usage.get("total_tokens")

    def remember(self, provider_id: str, cred: Credential) -> None:
        self.credentials[provider_id] = cred

    async def apply(self, provider: Provider | str, cred: Credential) -> None:
        if isinstance(provider, str):
            p = get_provider(provider)
            if p is None:
                raise ValueError(f"unknown provider {provider!r}")
            provider = p
        old, old_route = self.client, self.route
        self.route = None
        self.client = provider.make_client(cred)
        self.provider_id, self.cred_source, self.credential = provider.id, cred.source, cred
        self.credentials[provider.id] = cred
        if old_route is not None:
            await old_route.close()
        else:
            await _close_client(old)

    def use_route_target(self, target: RouteTarget) -> None:
        """Point status and compaction at the target that actually answered."""
        self.client = target.client
        self.provider_id = target.provider_id
        self.cred_source = target.credential.source
        self.credential = target.credential
        self.credentials[target.provider_id] = target.credential
        self.active_model_ref = target.ref
        self.context_limit = target.context_limit

    async def apply_route(self, route: FailoverRoute) -> None:
        old, old_route = self.client, self.route
        self.route = route
        self.model = self.model_ref = AUTO_REF
        self.use_route_target(route.current)
        if old_route is not None:
            await old_route.close()
        else:
            await _close_client(old)

    async def close(self) -> None:
        if self.route is not None:
            await self.route.close()
        else:
            await _close_client(self.client)

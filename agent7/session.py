"""REPL / oneshot session state."""

from __future__ import annotations

import asyncio
import os
import re
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent0_runtime import DEFAULT_MAX_TURNS
from agent0_runtime.history import estimate_tokens
from agent0_runtime.lifecycle import LifecycleHooks
from openai import AsyncOpenAI

from .providers import Credential, Provider, get_provider
from .routing import AUTO_REF, FailoverRoute, RouteTarget
from .store import new_session_id, timestamp


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
    # /models reports 1M for kimi-for-coding but 256k for its highspeed variant.
    if "kimi-for-coding" in key:
        return 262_144 if "highspeed" in key else 1_048_576
    if key == "k3" or key.startswith("k3.1"):
        return 1_048_576
    return None


def fmt_tokens(n: int) -> str:
    if n < 10_000:
        return f"{n:,}"
    k = n / 1000
    return f"{k:.0f}k" if k >= 100 or abs(k - round(k)) < 0.05 else f"{k:.1f}k"


def fmt_rounds(n: int) -> str:
    return f"{n} round" if n == 1 else f"{n} rounds"


def fmt_limit(n: int) -> str:
    """A context limit in the unit its vendor prints: 1M, 256k, 128k.

    A model sold as 1M has 1,048,576 tokens, so the decimal formatter used for
    real token counts would print 1049k — the same window under a different
    base. Match the label instead: pick whichever base divides the number
    exactly, so 1,048,576 → 1M, 262,144 → 256k, 256,000 → 256k.
    """
    for suffix, base in (("M", 1_000_000), ("M", 1024**2), ("k", 1_000), ("k", 1024)):
        if n % base == 0 and 1 <= n // base < 1000:
            return f"{n // base}{suffix}"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    return fmt_tokens(n)


def format_context_prompt(session: Session) -> str:
    u = fmt_tokens(session.context_tokens)
    return u if session.context_limit is None else f"{u}/{fmt_limit(session.context_limit)}"


def context_pressure(session: Session) -> int | None:
    """Share of the context window in use, as a percentage, or None without a limit."""
    if not session.context_limit:
        return None
    return min(100, round(100 * session.context_tokens / session.context_limit))


def fmt_cwd(path: Path, *, limit: int = 44) -> str:
    """The workspace as `~`-relative text, shortened from the left when long."""
    text, home = str(path), str(Path.home())
    if text == home:
        text = "~"
    elif text.startswith(home.rstrip("/") + "/"):
        text = "~" + text[len(home) :]
    if limit > 1 and len(text) > limit:
        tail = text[1 - limit :]
        cut = tail.find("/")
        text = "…" + (tail[cut:] if 0 <= cut < len(tail) - 1 else tail)
    return text


def format_usage(session: Session) -> str:
    """Context occupancy now, plus what this run and this session actually spent.

    ``context_tokens`` is the last reported round, or an estimate for a
    conversation that was resumed before any turn ran; the status bar shows the
    same number without the ``~`` estimate marker.
    """

    context = format_context_prompt(session)
    parts = [f"context~{context}" if session.context_estimated else f"context={context}"]
    if (pct := context_pressure(session)) is not None:
        parts[0] += f" ({pct}%)"
    if (spent := session.run_tokens.total) is not None:
        parts.append(f"run={fmt_tokens(spent)} ({fmt_rounds(session.run_tokens.rounds)})")
    if (spent := session.session_tokens.total) is not None:
        totals = session.session_tokens
        parts.append(
            f"session={fmt_tokens(spent)} "
            f"(prompt={fmt_tokens(totals.prompt_tokens or 0)} "
            f"completion={fmt_tokens(totals.completion_tokens or 0)})"
        )
    elif not session.has_turns:
        parts.append("session=0")
    else:
        # Nothing was reported for this conversation: say so instead of guessing.
        parts.append("session=unknown")
    return "  ".join(parts)


@dataclass
class TokenTotals:
    """Tokens a provider actually billed, summed over model rounds.

    Every field stays ``None`` until a provider reports it, and ``rounds`` counts
    the rounds that did, so "no usage" stays distinguishable from "zero tokens".
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    rounds: int = 0

    @property
    def known(self) -> bool:
        return self.rounds > 0

    @classmethod
    def from_dict(cls, data: Any) -> TokenTotals:
        """Rebuild totals from a snapshot, ignoring anything malformed."""

        totals = cls()
        if not isinstance(data, dict):
            return totals
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "rounds"):
            value = data.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(totals, name, value)
        return totals

    @property
    def total(self) -> int | None:
        """The reported total, or prompt+completion for providers that omit it."""

        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def add(self, usage: dict[str, int] | None, *, rounds: int = 1) -> None:
        if not usage:
            return
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, name, (getattr(self, name) or 0) + value)
        self.rounds += max(1, rounds)

    def reset(self) -> None:
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.rounds = 0


@dataclass
class Session:
    model: str
    cwd: Path
    client: AsyncOpenAI
    max_turns: int = DEFAULT_MAX_TURNS
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Host-owned base prompt; the runtime writes it into messages[0] at run time
    # together with any plugin fragments. Never persisted: it is rebuilt on load.
    system_prompt: str | None = None
    id: str = field(default_factory=new_session_id)
    name: str = ""
    created_at: str = field(default_factory=timestamp)
    persisted: bool = field(default=False, repr=False)
    models_cache: list[str] | None = None
    _models_fetch: Any = field(default=None, repr=False)
    _models_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    model_ref: str | None = None
    context_limit: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    run_tokens: TokenTotals = field(default_factory=TokenTotals)
    session_tokens: TokenTotals = field(default_factory=TokenTotals)
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
        """Track the last model round: these numbers size the next context."""

        self.prompt_tokens = usage.get("prompt_tokens")
        self.completion_tokens = usage.get("completion_tokens")
        self.total_tokens = usage.get("total_tokens")

    def record_run_usage(self, usage: dict[str, int] | None, *, rounds: int = 1) -> None:
        """Record what a finished run spent, on top of the session total.

        ``RunResult.usage`` is already summed over the run's model rounds, so it
        is added once; ``run_tokens`` holds that run and ``session_tokens`` keeps
        the whole conversation, which is what cost reporting needs.
        """

        self.run_tokens.reset()
        self.run_tokens.add(usage, rounds=rounds)
        self.session_tokens.add(usage, rounds=rounds)

    def reset_usage(self) -> None:
        """Forget usage that belongs to a conversation this session no longer holds."""

        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.run_tokens.reset()
        self.session_tokens.reset()

    @property
    def spent_tokens(self) -> int | None:
        """Tokens consumed in this session, for the docked status bar."""

        if (total := self.session_tokens.total) is not None:
            return total
        return self.run_tokens.total

    @property
    def has_turns(self) -> bool:
        """Whether the conversation holds anything beyond its system prompt."""

        return any(message.get("role") != "system" for message in self.messages)

    @property
    def context_tokens(self) -> int:
        """What the next request carries: the last round, else an estimate.

        A fresh session is exactly zero; a resumed conversation has no reported
        usage until its first turn, so its messages are estimated instead of
        showing an unhelpful ``?``.
        """

        if self.prompt_tokens is not None:
            return self.prompt_tokens
        if not self.has_turns:
            return 0
        return estimate_tokens(self.messages)

    @property
    def context_estimated(self) -> bool:
        return self.prompt_tokens is None and self.has_turns

    def restore_usage(self, block: Any) -> None:
        """Reload the usage stored with a saved conversation.

        Snapshots written before usage was recorded simply have no block; the
        context gauge then estimates from the messages and spend reads unknown.
        """

        self.reset_usage()
        if not isinstance(block, dict):
            return
        last = block.get("last")
        if isinstance(last, dict):
            self.update_usage(last)
        self.session_tokens = TokenTotals.from_dict(block.get("session"))

    def remember(self, provider_id: str, cred: Credential) -> None:
        self.credentials[provider_id] = cred

    def publish_models(
        self,
        fetch: Any,
        refs: list[str],
        credentials: dict[str, Credential],
    ) -> None:
        """Publish discovery only when it is still the session's current fetch."""
        with self._models_lock:
            if self._models_fetch is not fetch:
                return
            self.models_cache = refs
            self.credentials.update(credentials)

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
        # Model discovery runs in a daemon thread. Closing the UI must never
        # join it: a blocked DNS or HTTP call there must not delay Ctrl-D.
        with self._models_lock:
            self._models_fetch = None
        if self.route is not None:
            await self.route.close()
        else:
            await _close_client(self.client)

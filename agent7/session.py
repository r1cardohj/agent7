"""REPL / oneshot session state."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .providers import Credential, Provider, get_provider

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
    if "deepseek" in key:
        return 1_048_576 if "flash" in key else 128_000
    if key.startswith(("k3", "kimi")):
        return 1_048_576
    if "moonshot" in key:
        return 128_000
    return None

def fmt_tokens(n: int) -> str:
    if n < 10_000:
        return f"{n:,}"
    k = n / 1000
    return f"{k:.0f}k" if k >= 100 or abs(k - round(k)) < 0.05 else f"{k:.1f}k"

def format_context_prompt(session: Session) -> str:
    u = fmt_tokens(session.prompt_tokens) if session.prompt_tokens is not None else "?"
    return u if session.context_limit is None else f"{u}/{fmt_tokens(session.context_limit)}"

def format_usage(session: Session) -> str:
    if session.prompt_tokens is None:
        return "usage=none" + (
            f"  context_limit={fmt_tokens(session.context_limit)}" if session.context_limit else ""
        )
    parts = [
        f"prompt_tokens={fmt_tokens(session.prompt_tokens)}",
        f"completion_tokens={fmt_tokens(session.completion_tokens or 0)}",
    ]
    if session.total_tokens is not None:
        parts.append(f"total_tokens={fmt_tokens(session.total_tokens)}")
    if session.context_limit is not None:
        pct = min(100, round(100 * session.prompt_tokens / session.context_limit))
        parts.append(f"context_limit={fmt_tokens(session.context_limit)} ({pct}%)")
    return "  ".join(parts)

@dataclass
class Session:
    model: str
    cwd: Path
    client: AsyncOpenAI
    max_turns: int = 30
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

    def set_model(self, model: str, *, ref: str | None = None) -> None:
        self.model = model
        self.model_ref = ref or (f"{self.provider_id}/{model}" if self.provider_id else model)
        self.context_limit = resolve_context_limit(model)

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
        old, self.client = self.client, provider.make_client(cred)
        self.provider_id, self.cred_source, self.credential = provider.id, cred.source, cred
        self.credentials[provider.id] = cred
        with suppress(Exception):
            await old.close()

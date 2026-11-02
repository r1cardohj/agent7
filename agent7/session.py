"""REPL / oneshot session state."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# Hardcoded — DeepSeek /models has no context_length field.
_KNOWN_CONTEXT: dict[str, int] = {
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 128_000,
    "deepseek-flash": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4.1-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
}


def resolve_context_limit(model: str) -> int | None:
    raw = os.environ.get("AGENT7_CONTEXT_LIMIT", "").strip()
    if raw:
        try:
            return int(raw.replace("_", "").replace(",", ""))
        except ValueError:
            pass
    key = model.lower().strip()
    if key in _KNOWN_CONTEXT:
        return _KNOWN_CONTEXT[key]
    m = re.search(r"(?<![a-z0-9])(\d+)\s*k(?![a-z0-9])", key)
    if m:
        return int(m.group(1)) * 1000
    if "deepseek" in key:
        return 1_048_576 if "flash" in key else 128_000
    if "moonshot" in key or key.startswith("kimi"):
        return 128_000
    return None


def fmt_tokens(n: int) -> str:
    if n >= 10_000:
        k = n / 1000
        return f"{k:.0f}k" if k >= 100 or abs(k - round(k)) < 0.05 else f"{k:.1f}k"
    return f"{n:,}"


def format_context_prompt(session: Session) -> str:
    """REPL prefix: last usage.prompt_tokens / known context limit."""
    used = session.prompt_tokens
    limit = session.context_limit
    u = fmt_tokens(used) if used is not None else "?"
    if limit is None:
        return u
    return f"{u}/{fmt_tokens(limit)}"


def format_usage(session: Session) -> str:
    """Render last API usage fields as returned."""
    if session.prompt_tokens is None:
        limit = (
            f"  context_limit={fmt_tokens(session.context_limit)}"
            if session.context_limit
            else ""
        )
        return f"usage=none{limit}"
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
    context_limit: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def set_model(self, model: str) -> None:
        self.model = model
        self.context_limit = resolve_context_limit(model)

    def update_usage(self, usage: dict[str, int]) -> None:
        self.prompt_tokens = usage.get("prompt_tokens")
        self.completion_tokens = usage.get("completion_tokens")
        self.total_tokens = usage.get("total_tokens")

"""Slash-command interface for the REPL."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .session import Session, fmt_tokens, format_usage


@dataclass
class CommandResult:
    message: str = ""


SlashHandler = Callable[[Session, list[str]], CommandResult | Awaitable[CommandResult]]

# name → (help, handler)
_COMMANDS: dict[str, tuple[str, SlashHandler]] = {}


def command(name: str, *, help: str) -> Callable[[SlashHandler], SlashHandler]:
    def deco(fn: SlashHandler) -> SlashHandler:
        _COMMANDS[name.lower().lstrip("/")] = (help, fn)
        return fn

    return deco


def list_commands() -> list[tuple[str, str]]:
    return sorted((n, h) for n, (h, _) in _COMMANDS.items())


def known_names() -> list[str]:
    return sorted(_COMMANDS)


async def fetch_models(session: Session, *, refresh: bool = False) -> list[str]:
    if session.models_cache is not None and not refresh:
        return session.models_cache
    page = await session.client.models.list()
    ids = sorted({m.id for m in page.data if getattr(m, "id", None)})
    session.models_cache = ids
    return ids


async def dispatch(session: Session, line: str) -> CommandResult | None:
    raw = line.strip()
    if not raw.startswith("/"):
        return None
    body = raw[1:].strip()
    if not body:
        return CommandResult(message="empty command — try /help")
    name, _, rest = body.partition(" ")
    key = name.lower()
    args = rest.split() if rest.strip() else []
    entry = _COMMANDS.get(key)
    if not entry:
        return CommandResult(message=f"unknown command /{key}  (see /help)")
    _help, handler = entry
    result = handler(session, args)
    if asyncio.iscoroutine(result):
        result = await result
    return result


@command("help", help="list slash commands")
def cmd_help(session: Session, args: list[str]) -> CommandResult:
    lines = ["commands:"]
    for name, help_text in list_commands():
        lines.append(f"  /{name:<12} {help_text}")
    lines.append(f"current: model={session.model}  cwd={session.cwd}")
    return CommandResult(message="\n".join(lines))


@command("model", help="list/switch models  (/model | /model <id> | /model refresh)")
async def cmd_model(session: Session, args: list[str]) -> CommandResult:
    refresh = bool(args and args[0].lower() in ("refresh", "-r"))
    try:
        available = await fetch_models(session, refresh=refresh)
    except Exception as e:  # noqa: BLE001
        return CommandResult(message=f"failed to fetch models: {e}")

    if refresh and not args[1:]:
        return CommandResult(
            message="models:\n" + "\n".join(f"  {m}" for m in available)
        )

    if not args or refresh:
        lines = [f"model={session.model}", "available:"]
        for mid in available:
            mark = "*" if mid == session.model else " "
            lines.append(f"  {mark} {mid}")
        lines.append("usage: /model <id>  |  /model refresh")
        return CommandResult(message="\n".join(lines))

    name = args[0]
    lower = {m.lower(): m for m in available}
    resolved = lower.get(name.lower(), name)
    old = session.model
    session.set_model(resolved)
    note = ""
    if available and resolved not in available:
        note = f"\nwarning: {resolved!r} not in /models — using anyway"
    limit = (
        f"  context={fmt_tokens(session.context_limit)}"
        if session.context_limit
        else ""
    )
    return CommandResult(message=f"model: {old} → {resolved}{limit}{note}")


@command("status", help="show session status")
def cmd_status(session: Session, args: list[str]) -> CommandResult:
    n = max(0, len(session.messages) - 1)
    return CommandResult(
        message=(
            f"model={session.model}\n"
            f"cwd={session.cwd}\n"
            f"messages={n}  max_turns={session.max_turns}\n"
            f"{format_usage(session)}\n"
            f"base_url={session.client.base_url}"
        )
    )

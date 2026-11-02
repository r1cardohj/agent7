"""Slash-command interface for the REPL."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from getpass import getpass

from .agent import compact_messages
from .providers import (
    agent7_auth_path,
    describe_found,
    get_provider,
    list_providers,
    load_auth,
    resolve_all,
)
from .session import Session, fmt_tokens, format_usage

@dataclass
class CommandResult:
    message: str = ""

SlashHandler = Callable[[Session, list[str]], CommandResult | Awaitable[CommandResult]]
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
    if session.credential and session.provider_id:
        session.remember(session.provider_id, session.credential)
    for pid, cred in resolve_all().items():
        session.credentials.setdefault(pid, cred)
    if not session.credentials:
        session.models_cache = []
        return []
    order = {p.id: i for i, p in enumerate(list_providers())}
    errors, refs = [], []

    async def _one(pid: str):
        p = get_provider(pid)
        if not p:
            return pid, [], "unknown"
        try:
            return pid, await p.list_models(session.credentials[pid]), None
        except Exception as e:  # noqa: BLE001
            return pid, [], str(e)

    for pid, ids, err in await asyncio.gather(*[_one(p) for p in session.credentials]):
        if err:
            errors.append(f"{pid}: {err}")
        else:
            refs.extend(f"{pid}/{m}" for m in ids)
    refs = sorted(set(refs), key=lambda r: (order.get(r.partition("/")[0], 99), r))
    session.models_cache = refs
    if errors and not refs:
        raise RuntimeError("; ".join(errors))
    return refs

def resolve_model_ref(session: Session, name: str) -> tuple[str, str, str] | str:
    available, raw = session.models_cache or [], name.strip()
    lower = {m.lower(): m for m in available}
    if "/" in raw:
        ref = lower.get(raw.lower())
        if ref is None:
            prov, _, mid = raw.partition("/")
            return (prov, mid, f"{prov}/{mid}") if mid else f"bad model ref: {name!r}"
        prov, _, mid = ref.partition("/")
        return prov, mid, ref
    matches = [m for m in available if m.partition("/")[2].lower() == raw.lower()]
    if len(matches) == 1:
        ref = matches[0]
        return ref.partition("/")[0], ref.partition("/")[2], ref
    if len(matches) > 1:
        return f"ambiguous model {raw!r} — pick one of: {', '.join(matches)}"
    if raw.lower().startswith("deepseek"):
        return "deepseek", raw, f"deepseek/{raw}"
    if session.provider_id:
        return session.provider_id, raw, f"{session.provider_id}/{raw}"
    return f"unknown model {raw!r} — try /model"

async def switch_model(session: Session, name: str) -> CommandResult:
    resolved = resolve_model_ref(session, name)
    if isinstance(resolved, str):
        return CommandResult(message=resolved)
    provider_id, model_id, ref = resolved
    p = get_provider(provider_id)
    if p is None:
        return CommandResult(message=f"unknown provider {provider_id!r}")
    cred = session.credentials.get(provider_id) or p.resolve()
    if cred is None:
        alias = p.aliases[0] if p.aliases else p.id
        return CommandResult(message=f"no credentials for {provider_id} — /login {alias}")
    session.remember(provider_id, cred)
    if session.provider_id != provider_id or session.credential is not cred:
        await session.apply(p, cred)
    old = session.model_ref or session.model
    session.set_model(model_id, ref=ref)
    limit = f"  context={fmt_tokens(session.context_limit)}" if session.context_limit else ""
    return CommandResult(message=f"model: {old} → {ref}{limit}")

async def dispatch(session: Session, line: str) -> CommandResult | None:
    raw = line.strip()
    if not raw.startswith("/"):
        return None
    body = raw[1:].strip()
    if not body:
        return CommandResult(message="empty command — try /help")
    name, _, rest = body.partition(" ")
    entry = _COMMANDS.get(name.lower())
    if not entry:
        return CommandResult(message=f"unknown command /{name.lower()}  (see /help)")
    result = entry[1](session, rest.split() if rest.strip() else [])
    return await result if asyncio.iscoroutine(result) else result

async def _activate(session: Session, prefer: str | None = None) -> CommandResult:
    session.credentials.update(resolve_all())
    target = get_provider(prefer) if prefer else None
    cred = (session.credentials.get(target.id) or target.resolve()) if target else None
    if target and cred:
        session.remember(target.id, cred)
    if not target or not cred:
        if session.credential and session.provider_id:
            target, cred = get_provider(session.provider_id), session.credential
        elif session.credentials:
            pid = next(iter(session.credentials))
            target, cred = get_provider(pid), session.credentials[pid]
    if not target or not cred:
        return CommandResult(message="no credentials — try /login deepseek | /login kimi")
    await session.apply(target, cred)
    try:
        ids, note = await fetch_models(session, refresh=True), ""
    except Exception as e:  # noqa: BLE001
        ids, note = [], f"\nmodels fetch failed: {e}"
    cur = session.model_ref or session.model
    if ids and (not session.model or session.model == "pending" or cur not in ids):
        pick = next((r for r in ids if r.startswith(f"{target.id}/")), ids[0])
        await switch_model(session, pick)
    return CommandResult(
        message=(
            f"active: {target.id}  source={cred.source}\n"
            f"providers: [{', '.join(session.credentials)}]\n"
            f"base_url={cred.base_url}\nmodel={session.model_ref or session.model}{note}"
        )
    )

@command("help", help="list slash commands")
def cmd_help(session: Session, args: list[str]) -> CommandResult:
    lines = ["commands:"] + [f"  /{n:<12} {h}" for n, h in list_commands()]
    lines.append(f"current: model={session.model_ref or session.model}  cwd={session.cwd}")
    lines.append("tip: type @ in the prompt to complete a file path")
    return CommandResult(message="\n".join(lines))

@command("model", help="list/switch models  (/model | /model <ref> | /model refresh)")
async def cmd_model(session: Session, args: list[str]) -> CommandResult:
    refresh = bool(args and args[0].lower() in ("refresh", "-r"))
    try:
        available = await fetch_models(session, refresh=refresh)
    except Exception as e:  # noqa: BLE001
        return CommandResult(message=f"failed to fetch models: {e}")
    cur = session.model_ref or session.model
    if not args or (refresh and len(args) == 1):
        providers = ", ".join(session.credentials) or (session.provider_id or "?")
        lines = [f"model={cur}  providers=[{providers}]", "available:"]
        lines += [f"  {'*' if m == cur else ' '} {m}" for m in available]
        lines.append("usage: /model <id|provider/id>  |  /model refresh")
        return CommandResult(message="\n".join(lines))
    return await switch_model(session, args[1] if refresh else args[0])

@command("status", help="show session status")
def cmd_status(session: Session, args: list[str]) -> CommandResult:
    n = max(0, len(session.messages) - 1)
    return CommandResult(message=(
        f"model={session.model_ref or session.model}\n"
        f"provider={session.provider_id}  source={session.cred_source}\n"
        f"providers=[{', '.join(session.credentials) or '?'}]\n"
        f"cwd={session.cwd}\nmessages={n}  max_turns={session.max_turns}\n"
        f"{format_usage(session)}\nbase_url={session.client.base_url}"
    ))

@command("login", help="paste API key  (/login deepseek | /login kimi [global])")
async def cmd_login(session: Session, args: list[str]) -> CommandResult:
    if not args:
        rows = describe_found()
        names = " | ".join(p.id for p in list_providers())
        lines = [f"usage: /login <provider>   ({names})", f"auth file: {agent7_auth_path()}"]
        lines += (["found:"] + [f"  {a}  {b}" for a, b in rows]) if rows else ["found: (none)"]
        if session.provider_id:
            lines.append(f"active: {session.provider_id}  source={session.cred_source}")
        return CommandResult(message="\n".join(lines))
    target = get_provider(args[0])
    if not target:
        return CommandResult(message=f"unknown provider {args[0]!r}")
    try:
        cred = await target.login(args=args[1:], prompt_secret=lambda s: getpass(f"{s}: ").strip())
    except Exception as e:  # noqa: BLE001
        return CommandResult(message=f"{target.id} login failed: {e}")
    session.remember(target.id, cred)
    result = await _activate(session, target.id)
    return CommandResult(message=f"logged in {target.id}\n{result.message}")

@command("compact", help="shrink old tool/assistant text in history")
def cmd_compact(session: Session, args: list[str]) -> CommandResult:
    new_msgs, n = compact_messages(session.messages)
    session.messages[:] = new_msgs
    if not n:
        return CommandResult(message="compact: nothing to shrink (history short or already small)")
    return CommandResult(message=f"compact: shrunk {n} message(s); recent turns kept intact")

@command("logout", help="clear saved key  (/logout deepseek|kimi|moonshot)")
def cmd_logout(session: Session, args: list[str]) -> CommandResult:
    if not args:
        keys = [p.id for p in list_providers() if p.id in load_auth()]
        return CommandResult(
            message="nothing stored" if not keys else "usage: /logout <provider>\nstored: " + ", ".join(keys)
        )
    p = get_provider(args[0])
    if not p:
        return CommandResult(message=f"unknown provider {args[0]!r}")
    if p.logout():
        session.credentials.pop(p.id, None)
        return CommandResult(message=f"cleared {p.id}")
    return CommandResult(message=f"no stored key for {p.id}")

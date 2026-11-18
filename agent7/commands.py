"""Slash-command interface for the REPL."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from getpass import getpass

from agent0_runtime.history import compact_messages

from .config import SYSTEM
from .providers import (
    Credential,
    agent7_auth_path,
    describe_found,
    get_provider,
    list_providers,
    load_auth,
    login_hints,
    resolve_all,
)
from .routing import AUTO_REF, build_auto_route
from .session import Session, fmt_limit, format_usage, resolve_context_limit
from .store import delete_session, list_sessions, load, resume_messages


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


MODEL_LIST_TIMEOUT = 5.0
MODEL_LIST_JOIN_TIMEOUT = 12.0


def _discover_models(session: Session) -> tuple[list[str], dict[str, Credential]]:
    """Synchronous provider sweep for a daemon worker, bounded per provider."""
    credentials = dict(session.credentials)
    if session.credential and session.provider_id:
        credentials[session.provider_id] = session.credential
    for pid, cred in resolve_all().items():
        credentials.setdefault(pid, cred)
    if not credentials:
        return [], credentials
    order = {p.id: i for i, p in enumerate(list_providers())}
    errors, refs = [], []
    for pid, cred in credentials.items():
        p = get_provider(pid)
        if not p:
            errors.append(f"{pid}: unknown")
            continue
        try:
            ids = p.list_models_sync(cred, timeout=MODEL_LIST_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pid}: {e}")
        else:
            refs.extend(f"{pid}/{m}" for m in ids)
    refs = sorted(set(refs), key=lambda r: (order.get(r.partition("/")[0], 99), r))
    if errors and not refs:
        raise RuntimeError("; ".join(errors))
    return refs, credentials


class ModelsFetch:
    """Daemon-thread provider discovery that process exit never waits for."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.error: BaseException | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="agent7-model-discovery",
            daemon=True,
        )

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> None:
        self._thread.start()

    async def wait(self, timeout: float | None = None) -> None:
        """Cooperatively join from the UI loop without owning the worker thread."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not self._done.is_set():
            if deadline is not None and loop.time() >= deadline:
                raise TimeoutError("model list is still loading")
            await asyncio.sleep(0.05)
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        try:
            refs, credentials = _discover_models(self.session)
            self.session.publish_models(self, refs, credentials)
        except BaseException as e:  # noqa: BLE001  background failure surfaces on /model
            self.error = e
        finally:
            self._done.set()


def _start_models_fetch(session: Session) -> ModelsFetch:
    fetch = ModelsFetch(session)
    session._models_fetch = fetch
    fetch.start()
    return fetch


def prefetch_models(session: Session) -> ModelsFetch | None:
    """Populate the model cache without putting network work on the UI loop."""
    if session.models_cache is not None:
        return None
    fetch = session._models_fetch
    if fetch is None or fetch.done:
        fetch = _start_models_fetch(session)
    return fetch


async def fetch_models(session: Session, *, refresh: bool = False) -> list[str]:
    fetch = session._models_fetch
    if not refresh and fetch is not None and not fetch.done:
        await fetch.wait(MODEL_LIST_JOIN_TIMEOUT)
    if not refresh and session.models_cache is not None:
        return session.models_cache
    if fetch is not None and fetch.done and fetch.error is not None:
        raise fetch.error
    fetch = _start_models_fetch(session)
    await fetch.wait(MODEL_LIST_JOIN_TIMEOUT)
    if session.models_cache is not None:
        return session.models_cache
    raise RuntimeError("model list unavailable")


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
    for p in list_providers():
        if any(raw.lower().startswith(n) for n in (p.id, *p.aliases)):
            return p.id, raw, f"{p.id}/{raw}"
    if session.provider_id:
        return session.provider_id, raw, f"{session.provider_id}/{raw}"
    return f"unknown model {raw!r} — try /model"


async def switch_model(session: Session, name: str) -> CommandResult:
    if name.strip().lower() == AUTO_REF:
        try:
            route = build_auto_route(
                session.credentials,
                models=session.models_cache,
                context_limit=resolve_context_limit,
            )
        except ValueError as e:
            return CommandResult(message=str(e))
        old = session.model_ref or session.model
        await session.apply_route(route)
        return CommandResult(message=f"model: {old} → {session.display_model}")
    resolved = resolve_model_ref(session, name)
    if isinstance(resolved, str):
        return CommandResult(message=resolved)
    provider_id, model_id, ref = resolved
    p = get_provider(provider_id)
    if p is None:
        return CommandResult(message=f"unknown provider {provider_id!r}")
    cred = session.credentials.get(provider_id) or p.resolve()
    if cred is None:
        return CommandResult(
            message=f"no credentials for {provider_id} — /login {p.preferred_login}"
        )
    session.remember(provider_id, cred)
    if (
        session.route is not None
        or session.provider_id != provider_id
        or session.credential is not cred
    ):
        await session.apply(p, cred)
    old = session.model_ref or session.model
    session.set_model(model_id, ref=ref)
    limit = f"  context={fmt_limit(session.context_limit)}" if session.context_limit else ""
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
    keep_auto = session.model_ref == AUTO_REF
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
        return CommandResult(message=f"no credentials — try {login_hints()}")
    if not keep_auto:
        await session.apply(target, cred)
    try:
        ids, note = await fetch_models(session, refresh=True), ""
    except Exception as e:  # noqa: BLE001
        ids, note = [], f"\nmodels fetch failed: {e}"
    cur = session.model_ref or session.model
    if keep_auto:
        result = await switch_model(session, AUTO_REF)
        if result.message.startswith("auto:"):
            note += f"\n{result.message}"
    elif ids and (not session.model or session.model == "pending" or cur not in ids):
        pick = next((r for r in ids if r.startswith(f"{target.id}/")), ids[0])
        await switch_model(session, pick)
    return CommandResult(
        message=(
            f"active: {target.id}  source={cred.source}\n"
            f"providers: [{', '.join(session.credentials)}]\n"
            f"base_url={cred.base_url}\nmodel={session.display_model}{note}"
        )
    )


@command("help", help="list slash commands")
def cmd_help(session: Session, _args: list[str]) -> CommandResult:
    lines = ["commands:"] + [f"  /{n:<12} {h}" for n, h in list_commands()]
    lines.append(f"current: model={session.display_model}  cwd={session.cwd}")
    lines.append("tip: @, /cmd and /model args are fuzzy-matched (try @mn, /mdl)")
    return CommandResult(message="\n".join(lines))


@command("model", help="list/switch models  (/model | /model <ref> | /model refresh)")
async def cmd_model(session: Session, args: list[str]) -> CommandResult:
    refresh = bool(args and args[0].lower() in ("refresh", "-r"))
    try:
        available = list(await fetch_models(session, refresh=refresh))
        if session.credentials:
            available.append(AUTO_REF)
    except Exception as e:  # noqa: BLE001
        return CommandResult(message=f"failed to fetch models: {e}")
    cur = session.model_ref or session.model
    if not args or (refresh and len(args) == 1):
        providers = ", ".join(session.credentials) or (session.provider_id or "?")
        lines = [f"model={session.display_model}  providers=\\[{providers}]", "available:"]
        lines += [f"  {'*' if m == cur else ' '} {m}" for m in available]
        lines.append("usage: /model <id|provider/id>  |  /model refresh")
        return CommandResult(message="\n".join(lines))
    return await switch_model(session, args[1] if refresh else args[0])


@command("status", help="show session status")
def cmd_status(session: Session, _args: list[str]) -> CommandResult:
    n = max(0, len(session.messages) - 1)
    identity = ""
    if session_id := getattr(session, "id", ""):
        identity = f"session={getattr(session, 'name', '') or 'New session'} ({session_id})\n"
    route_info = ""
    if session.route and session.route.selection:
        selection = session.route.selection
        scores = ", ".join(
            f"{ref}={score:.0f}"
            for ref, score in sorted(selection.scores.items(), key=lambda item: -item[1])
        )
        route_info = f"\nscorer={selection.scorer_ref}\nscores={scores or 'unavailable'}"
        if selection.error:
            route_info += f"  error={selection.error}"
    return CommandResult(
        message=(
            f"model={session.display_model}\n{identity}"
            f"provider={session.provider_id}  source={session.cred_source}\n"
            f"providers=\\[{', '.join(session.credentials) or '?'}]\n"
            f"cwd={session.cwd}\nmessages={n}  max_turns={session.max_turns}\n"
            f"{format_usage(session)}\nbase_url={session.client.base_url}{route_info}"
        )
    )


async def _delete_session(session: Session, args: list[str]) -> CommandResult:
    from .repl import confirm_sessions_delete, pick_sessions

    current = getattr(session, "id", "")
    if args:
        session_ids = list(args)
    else:
        snapshots = [
            snapshot for snapshot in list_sessions(session.cwd) if snapshot.get("id") != current
        ]
        if not snapshots:
            return CommandResult(message="no other saved sessions in this workspace")
        picked = await pick_sessions(snapshots)
        if not picked:
            return CommandResult(
                message="delete: cancelled (Esc/Ctrl-C in the picker selects nothing)"
            )
        session_ids = picked
    if current in session_ids:
        return CommandResult(
            message="cannot delete the active session — switch to another session first"
        )
    try:
        snapshots = [load(session.cwd, session_id) for session_id in session_ids]
        if not confirm_sessions_delete(snapshots):
            kept = ", ".join(
                f"{snapshot.get('name') or 'New session'} ({snapshot.get('id', '?')})"
                for snapshot in snapshots
            )
            return CommandResult(message=f"delete: cancelled — kept {kept}")
        trashed = [
            (snapshot, delete_session(session.cwd, snapshot["id"])) for snapshot in snapshots
        ]
    except ValueError as e:
        return CommandResult(message=str(e))
    lines = [
        f"deleted {snapshot.get('name') or 'New session'} "
        f"({snapshot.get('id', '?')}) → {trash_path}"
        for snapshot, trash_path in trashed
    ]
    return CommandResult(message="\n".join(lines))


@command("session", help="switch/delete sessions  (/session | /session delete <id...>)")
async def cmd_session(session: Session, args: list[str]) -> CommandResult:
    # Picker/UI lives in repl; the local import avoids the commands <-> repl
    # import cycle while keeping this a normal discoverable slash command.
    from .repl import pick_session, print_history

    current = getattr(session, "id", "")
    if args and args[0].lower() in {"delete", "rm"}:
        return await _delete_session(session, args[1:])
    if args:
        session_id = args[0]
    else:
        snapshots = list_sessions(session.cwd)
        if not snapshots:
            return CommandResult(message="no saved sessions in this workspace")
        picked = await pick_session(snapshots, current=current)
        if picked is None:
            return CommandResult(message="session: cancelled")
        session_id = picked
    if session_id == current:
        return CommandResult(
            message=(
                f"session: already on {getattr(session, 'name', '') or 'New session'}"
                f" ({session_id})"
            )
        )
    try:
        snapshot = load(session.cwd, session_id)
    except ValueError as e:
        return CommandResult(message=str(e))

    saved_model = snapshot.get("model_ref")
    if saved_model and saved_model != (session.model_ref or session.model):
        result = await switch_model(session, saved_model)
        if result.message.startswith(
            ("unknown", "ambiguous", "auto:", "bad model", "no credentials")
        ):
            return CommandResult(message=f"session not switched: {result.message}")

    old = getattr(session, "name", "") or current
    system = SYSTEM
    if session.messages and session.messages[0].get("role") == "system":
        content = session.messages[0].get("content")
        if isinstance(content, str):
            system = content
    messages = resume_messages(snapshot, system)
    session.id = snapshot["id"]
    session.name = snapshot["name"]
    session.created_at = snapshot["created_at"]
    session.persisted = True
    session.messages[:] = messages
    # Usage belongs to the conversation that produced it, not to the REPL
    # process. The next turn will populate it from the selected session.
    session.reset_usage()
    print_history(snapshot["messages"])
    return CommandResult(message=f"session: {old} → {session.name} ({session.id})")


@command("login", help="paste API key  (/login deepseek | /login kimi)")
async def cmd_login(session: Session, args: list[str]) -> CommandResult:
    if not args:
        rows = describe_found()
        names = " | ".join(p.preferred_login for p in list_providers())
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
def cmd_compact(session: Session, _args: list[str]) -> CommandResult:
    new_msgs, n = compact_messages(session.messages)
    session.messages[:] = new_msgs
    if not n:
        return CommandResult(message="compact: nothing to shrink (history short or already small)")
    return CommandResult(message=f"compact: shrunk {n} message(s); recent turns kept intact")


@command("logout", help="clear saved key  (/logout deepseek | /logout kimi)")
def cmd_logout(session: Session, args: list[str]) -> CommandResult:
    if not args:
        keys = [p.id for p in list_providers() if p.id in load_auth()]
        return CommandResult(
            message="nothing stored"
            if not keys
            else "usage: /logout <provider>\nstored: " + ", ".join(keys)
        )
    p = get_provider(args[0])
    if not p:
        return CommandResult(message=f"unknown provider {args[0]!r}")
    if p.logout():
        session.credentials.pop(p.id, None)
        return CommandResult(message=f"cleared {p.id}")
    return CommandResult(message=f"no stored key for {p.id}")

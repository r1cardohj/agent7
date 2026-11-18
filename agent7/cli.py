"""Command line: options, session bootstrap, one-shot runs, REPL launch.

`agent7` is a click command; everything it needs to decide *what* to run lives
here, while `agent7.repl` owns the terminal (prompt, completion, printing).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from agent0_runtime import DEFAULT_MAX_TURNS
from agent0_runtime import AgentInterrupted as Interrupted
from agent0_runtime.lifecycle import emit_lifecycle

from . import __version__
from .commands import fetch_models, prefetch_models, switch_model
from .config import SHELL, system_prompt
from .herdr import HerdrReporter
from .providers import (
    default_model_for,
    get_provider,
    list_providers,
    login_hints,
    make_client,
    resolve_all,
    resolve_default,
)
from .repl import (
    confirm_sessions_delete,
    err,
    pick_model,
    pick_session,
    pick_sessions,
    print_history,
    repl_loop,
    run_turn,
)
from .routing import AUTO_REF
from .session import Session
from .store import (
    delete_session,
    list_sessions,
    new_session_id,
    resume,
    resume_messages,
    save,
    timestamp,
)

PROG = "agent7"
# What a bare `-m` parses to: pick the model for this run from a menu instead
# of naming it. No model ref is empty, so the sentinel cannot collide.
PICK_MODEL = ""
# A bare -r opens session discovery. Generated IDs are distinguishable from a
# legacy `-r "continue"` task, which remains a resume of the latest session.
PICK_SESSION = ""
# A bare --delete opens the same picker in removal mode.
PICK_DELETE = ""
RESUME_ID = re.compile(r"(?:sess-[0-9a-f]{32}|legacy-[0-9a-f]{16})")


def _resume_args(task: str, value: str | None) -> tuple[str, bool | str]:
    if value is None:
        return task, False
    if value == PICK_SESSION:
        return task, True
    if RESUME_ID.fullmatch(value):
        return task, value
    return " ".join(part for part in (value, task) if part), True


@dataclass(frozen=True)
class Options:
    """A parsed command line."""

    task: str = ""
    cwd: Path | None = None
    max_turns: int | None = None
    resume: bool | str = False  # True: latest/picker; a string: exact session ID
    model: str | None = None  # None: default; PICK_MODEL: ask; else the ref
    delete: str | None = None  # None: off; PICK_DELETE: picker; else exact ID

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Options:
        """Parse argv exactly like the CLI does; exits on --help or bad input."""
        try:
            ctx = main.make_context(PROG, list(sys.argv[1:] if argv is None else argv))
        except click.exceptions.Exit as exc:  # --help / --version already printed
            raise SystemExit(exc.exit_code) from None
        except click.UsageError as exc:
            exc.show()
            raise SystemExit(exc.exit_code) from None
        with ctx:
            params = dict(ctx.params)
        task, resume_value = _resume_args(
            " ".join(params.pop("task")).strip(), params.pop("resume")
        )
        return cls(task=task, resume=resume_value, **params)


def _positive_int(_ctx: click.Context, _param: click.Parameter, value: int | None) -> int | None:
    if value is not None and value < 1:
        raise click.BadParameter(f"must be at least 1, got {value}")
    return value


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def _force_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


class Agent7CLI(click.Command):
    """Top-level command with the reserved native `deploy` operation.

    Intercepting the first word before click parses top-level options lets
    `agent7 deploy --check` keep deploy-specific options without turning the
    natural-language task interface into a click command group.
    """

    def main(
        self,
        args: list[str] | tuple[str, ...] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: object,
    ) -> object:
        argv = list(sys.argv[1:] if args is None else args)
        if argv and argv[0] == "deploy":
            return deploy_command.main(
                args=argv[1:],
                prog_name=f"{PROG} deploy",
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        return super().main(
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            **extra,
        )


@click.command(
    cls=Agent7CLI,
    name=PROG,
    help="agent7 — minimal coding agent. Omit TASK to open the REPL.",
    epilog="Remote access: `agent7 deploy` prepares Tailscale + SSH for Herdr Mobile.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", prog_name=PROG, message=f"{PROG} %(version)s")
@click.argument("task", nargs=-1)
@click.option(
    "--cwd",
    type=click.Path(path_type=Path, file_okay=False),
    help="working directory (default: the current one)",
)
@click.option(
    "--max-turns",
    type=int,
    callback=_positive_int,
    default=None,
    metavar="N",
    help=f"model rounds per task (default {DEFAULT_MAX_TURNS}, or $AGENT7_MAX_TURNS)",
)
@click.option(
    "-m",
    "--model",
    "model",
    is_flag=False,
    flag_value=PICK_MODEL,
    default=None,
    metavar="REF",
    help="model for this run, e.g. deepseek-flash; a bare -m picks one from a menu "
    "(one-shot tasks otherwise default to auto)",
)
@click.option(
    "-r",
    "--resume",
    is_flag=False,
    flag_value=PICK_SESSION,
    default=None,
    metavar="SESSION_ID",
    help="resume a session by ID; bare -r picks one from a menu",
)
@click.option(
    "--delete",
    is_flag=False,
    flag_value=PICK_DELETE,
    default=None,
    metavar="SESSION_ID",
    help="move session(s) to trash; bare --delete multi-picks from a menu",
)
def main(
    task: tuple[str, ...],
    cwd: Path | None,
    max_turns: int | None,
    resume: str | None,
    delete: str | None,
    model: str | None,
) -> None:
    _force_utf8_stdio()
    _load_dotenv()
    if task and task[0] == "deploy":
        _run_deploy(list(task[1:]))
        return
    task_text, resume_value = _resume_args(" ".join(task).strip(), resume)
    options = Options(
        task=task_text,
        cwd=cwd,
        max_turns=max_turns,
        resume=resume_value,
        delete=delete,
        model=model,
    )
    try:
        asyncio.run(async_main(options))
    except KeyboardInterrupt:
        # Startup/cleanup happens outside the per-turn handler in repl_loop.
        err.print("\n[dim]# interrupted[/dim]")


@click.command(name="deploy")
@click.option(
    "--check", "check_only", is_flag=True, help="verify remote access without changing the system"
)
@click.option(
    "--stop", is_flag=True, help="stop tailscaled without removing its binaries or login state"
)
@click.option("--sshd", is_flag=True, help="use the system OpenSSH server instead of Tailscale SSH")
@click.option(
    "--auth-key",
    "auth_key",
    default=None,
    metavar="KEY",
    envvar="TAILSCALE_AUTHKEY",
    help="Tailscale auth key for unattended login (default: $TAILSCALE_AUTHKEY)",
)
def deploy_command(
    check_only: bool,
    stop: bool,
    sshd: bool,
    auth_key: str | None,
) -> None:
    """Prepare this machine for Herdr Mobile over Tailscale + SSH."""
    from .remote import RemoteError, deploy_mobile_ssh, stop_tailscale

    if stop and (check_only or sshd or auth_key):
        raise click.UsageError("--stop cannot be combined with --check, --sshd, or --auth-key")
    try:
        if stop:
            stop_tailscale()
            return
        code = deploy_mobile_ssh(check=check_only, sshd=sshd, auth_key=auth_key or "")
    except RemoteError as e:
        err.print(f"[red]# deploy failed: {e}[/red]")
        raise SystemExit(1) from None
    if code:
        raise SystemExit(code)


def _run_deploy(argv: list[str]) -> None:
    """Parse the reserved `agent7 deploy ...` words with the deploy command."""
    ctx = deploy_command.make_context("agent7 deploy", argv)
    with ctx:
        deploy_command.invoke(ctx)


def resolve_max_turns(options: Options) -> int:
    """CLI > $AGENT7_MAX_TURNS > default.

    A saved session keeps its transcript, not its turn budget: resuming must not
    pin you to whatever `--max-turns` the snapshot was created with.
    """
    if options.max_turns is not None:
        return options.max_turns
    raw = os.environ.get("AGENT7_MAX_TURNS", "").strip()
    if raw:
        try:
            if int(raw) >= 1:
                return int(raw)
        except ValueError:
            pass
        err.print(f"[dim]# ignoring bad AGENT7_MAX_TURNS={raw!r}[/dim]")
    return DEFAULT_MAX_TURNS


def can_pick_models() -> bool:
    """fzp's Picker needs a terminal on both ends; a pipe cannot choose."""
    return bool(sys.stdin and sys.stdin.isatty() and sys.stderr and sys.stderr.isatty())


def model_menu(session: Session, ids: list[str]) -> list[str]:
    """The refs a bare `-m` offers: models, provider defaults, and built-in auto."""
    refs = (
        list(ids)
        if ids
        else [
            f"{p.id}/{p.default_model}"
            for p in list_providers()
            if p.default_model and p.id in session.credentials
        ]
    )
    if session.credentials:
        refs.append(AUTO_REF)
    return refs


async def choose_model(session: Session, ids: list[str], *, current: str = "") -> str:
    """Pick this run's model from a fuzzy menu; exits instead of guessing."""
    items = model_menu(session, ids)
    if not items:
        err.print(f"[red]# no models to pick from — log in first: {login_hints()}[/red]")
        raise SystemExit(1)
    picked = await pick_model(items, current=current)
    if picked is None:
        err.print("[dim]# no model picked — cancelled[/dim]")
        raise SystemExit(1)
    return picked


async def delete_requested_session(options: Options, cwd: Path) -> None:
    """Pick/confirm one or more snapshots and soft-delete them before startup."""
    session_ids = [] if options.delete == PICK_DELETE else [options.delete]
    try:
        if options.delete == PICK_DELETE:
            if not can_pick_models():
                raise SystemExit(
                    "a bare --delete shows a session menu, which needs a terminal; "
                    "use agent7 --delete SESSION_ID instead"
                )
            snapshots = list_sessions(cwd)
            if not snapshots:
                raise SystemExit(f"no sessions for workspace: {cwd}")
            picked = await pick_sessions(snapshots)
            if not picked:
                err.print("[dim]# delete cancelled[/dim]")
                return
            session_ids = picked
        snapshots = [resume(cwd, session_id) for session_id in session_ids]
        if not confirm_sessions_delete(snapshots):
            kept = ", ".join(
                f"{snapshot.get('name') or 'New session'} ({snapshot.get('id', '?')})"
                for snapshot in snapshots
            )
            err.print(f"[dim]# delete cancelled — kept {kept}[/dim]")
            return
        trashed = [(snapshot, delete_session(cwd, snapshot["id"])) for snapshot in snapshots]
    except ValueError as e:
        raise SystemExit(str(e)) from None
    for snapshot, trash_path in trashed:
        err.print(
            f"[dim]# deleted {snapshot.get('name') or 'New session'} "
            f"({snapshot.get('id', '?')}) → {trash_path}[/dim]"
        )


async def async_main(options: Options) -> None:
    """Run one task, or open the REPL, for an already-parsed command line."""
    task = options.task or None
    cwd = (options.cwd or Path.cwd()).resolve()
    if options.delete is not None:
        if task or options.resume or options.model is not None or options.max_turns is not None:
            raise SystemExit("--delete cannot be combined with TASK, -r, -m, or --max-turns")
        await delete_requested_session(options, cwd)
        return
    saved = None
    resume_id = options.resume if isinstance(options.resume, str) else None
    choose_resume = options.resume is True and not task
    try:
        if choose_resume:
            if not can_pick_models():
                raise SystemExit(
                    "a bare -r shows a session menu, which needs a terminal; "
                    "use agent7 -r SESSION_ID instead"
                )
            snapshots = list_sessions(cwd)
            if not snapshots:
                raise SystemExit(f"no sessions for workspace: {cwd}")
            picked_session = await pick_session(snapshots)
            if picked_session is None:
                raise SystemExit("no session picked — cancelled")
            saved = resume(cwd, picked_session)
        else:
            # A legacy `-r "continue"` still means the latest session. Without
            # -r this read only supplies the last model for a fresh session.
            saved = resume(cwd, resume_id)
    except ValueError as e:
        if options.resume:
            raise SystemExit(str(e)) from None
    snapshot = saved if options.resume else None
    created_at = snapshot["created_at"] if snapshot else timestamp()
    max_turns = resolve_max_turns(options)
    # A bare `-m` asks for the menu. Only the option itself can carry that: an
    # empty model from $AGENT7_MODEL or the snapshot keeps its old meaning.
    pick = options.model == PICK_MODEL
    # `-m` > $AGENT7_MODEL > the workspace's last model.
    env_model = os.environ.get("AGENT7_MODEL", "").strip()
    workspace_model = env_model or (saved.get("model_ref", "") if saved else "")
    saved_provider, sep, _ = workspace_model.partition("/")
    reuse_saved_model = (
        saved is not None
        and options.model is None
        and (workspace_model == AUTO_REF or bool(sep) and get_provider(saved_provider) is not None)
        and not env_model
    )
    # A one-shot with no explicit choice starts on auto: the task itself is the
    # only hint about which model fits. `-m` and $AGENT7_MODEL still win, and
    # the saved model remains a fallback when auto has no usable candidates.
    implicit_auto = bool(task) and options.model is None and not env_model
    model = (
        AUTO_REF if implicit_auto else (workspace_model if options.model is None else options.model)
    )
    if pick and not can_pick_models():
        err.print(
            "[red]# a bare -m shows a menu, which needs a terminal — "
            "name the model instead: agent7 -m deepseek-flash[/red]"
        )
        raise SystemExit(1)

    all_creds = resolve_all()
    picked = resolve_default()
    if picked is None and all_creds:
        pid = next(iter(all_creds))
        prov = get_provider(pid)
        picked = (prov, all_creds[pid]) if prov else None
    if picked:
        prov, cred = picked
        client = prov.make_client(cred)
    else:
        client = make_client("missing", list_providers()[0].base_url)
        if task:
            keys = ", ".join(k for p in list_providers() for k in p.env_keys)
            print(
                f"No credentials. Set {keys},\nor run `uv run agent7` and `/login`.",
                file=sys.stderr,
            )
            sys.exit(1)
        prov, cred = None, None
    # Herdr-managed background runs are an optional optimization; ordinary bash
    # remains the fallback in every other environment. The prompt is host state
    # that the runtime writes into messages[0] on the first turn, so a resumed
    # conversation picks up the current one instead of the prompt it was saved with.
    state = Session(
        model=model or "pending",
        cwd=cwd,
        client=client,
        max_turns=max_turns,
        messages=resume_messages(snapshot) if snapshot else [],
        system_prompt=system_prompt(),
        id=snapshot["id"] if snapshot else new_session_id(),
        name=snapshot.get("name", "") if snapshot else "",
        created_at=created_at,
        persisted=snapshot is not None,
        provider_id=prov.id if prov else None,
        cred_source=cred.source if cred else None,
        credential=cred,
        credentials=dict(all_creds),
    )
    # A resumed conversation keeps the usage it was saved with, so the gauge and
    # /status describe the history on screen instead of starting from nothing.
    state.restore_usage(snapshot.get("usage") if snapshot else None)
    # A one-shot becomes a real session only once it runs: a run that never
    # starts (bad -m, menu cancelled) must not create a session snapshot.
    committed = snapshot is not None
    started = False
    state.hooks.subscribe(HerdrReporter())

    def persist(commit: bool = False) -> None:
        # Merely opening a new REPL must not create a picker entry. The first
        # agent turn makes it real; resumed/switched sessions are already real.
        nonlocal committed
        committed = committed or commit or state.persisted
        if committed:
            save(state, created_at=state.created_at)

    try:
        ids: list[str] = []
        # A saved provider/model ref is already actionable, and auto reaches its
        # reviewed targets without a list: do not make startup wait on every
        # provider's /models endpoint just to reuse one. Explicit choices and
        # first-time setup still fetch.
        if all_creds and not reuse_saved_model:
            try:
                ids = await fetch_models(state)
            except Exception as e:  # noqa: BLE001
                err.print(f"[dim]# models fetch failed: {e}[/dim]")
        if pick:
            model = await choose_model(state, ids, current=workspace_model)
        if model:
            result = await switch_model(state, model)
            if result.message.startswith(("unknown", "ambiguous", "auto:")):
                err.print(f"[dim]# {result.message}[/dim]")
                # A ref the user typed must not quietly become another model.
                # Neither may an explicit auto from $AGENT7_MODEL or a snapshot.
                if options.model is not None or (model == AUTO_REF and not implicit_auto):
                    raise SystemExit(1)
                # An implicit one-shot auto is only a default: try the saved
                # model, then the first fetched model, before falling back to
                # the active provider's default model.
                fallbacks = [workspace_model, *ids[:1]] if implicit_auto else ids[:1]
                for fallback in fallbacks:
                    if fallback and fallback != AUTO_REF:
                        result = await switch_model(state, fallback)
                        if not result.message.startswith(("unknown", "ambiguous", "auto:")):
                            break
                else:
                    if implicit_auto:
                        state.set_model(default_model_for(state.provider_id))
                    else:
                        state.set_model(model)
        elif state.model in ("", "pending"):
            if ids:
                await switch_model(state, ids[0])
            else:
                state.set_model(default_model_for(state.provider_id))
        started = True
        emit_lifecycle(state, "session_started")
        err.print(
            f"[dim]# agent7 v{__version__}  model={state.display_model}  "
            f"providers=\\[{','.join(state.credentials) or 'none'}]  "
            f"shell={SHELL.name}  cwd={state.cwd}[/dim]"
        )
        action = "resumed" if snapshot else "new"
        err.print(
            f"[dim]# {action} session  name={state.name or 'New session'}  id={state.id}[/dim]"
        )
        if snapshot:
            print_history(snapshot["messages"])
        if task:
            committed = True  # from here on the run owns its session snapshot
            try:
                await run_turn(state, task, on_saved=persist)
            except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
                err.print("\n[dim]# interrupted[/dim]")
            except Exception as e:  # noqa: BLE001
                err.print(f"[red]# error: {type(e).__name__}: {e}[/red]")
            return
        if state.models_cache is None:
            prefetch_models(state)
        await repl_loop(state, persist)
    finally:
        if started:
            emit_lifecycle(state, "session_ended")
        try:
            persist()
        finally:
            await state.close()

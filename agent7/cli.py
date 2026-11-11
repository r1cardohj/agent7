"""Command line: options, session bootstrap, one-shot runs, REPL launch.

`agent7` is a click command; everything it needs to decide *what* to run lives
here, while `agent7.repl` owns the terminal (prompt, completion, printing).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from openai import AsyncOpenAI

from . import __version__
from .agent import DEFAULT_MAX_TURNS, SHELL, SYSTEM, Interrupted
from .commands import fetch_models, switch_model
from .hooks import emit_lifecycle
from .providers import (
    default_model_for,
    get_provider,
    list_providers,
    login_hints,
    resolve_all,
    resolve_default,
)
from .repl import err, pick_model, print_history, repl_loop, run_turn
from .session import Session
from .store import resume, resume_messages, save, timestamp

PROG = "agent7"
# What a bare `-m` parses to: pick the model for this run from a menu instead
# of naming it. No model ref is empty, so the sentinel cannot collide.
PICK_MODEL = ""

@dataclass(frozen=True)
class Options:
    """A parsed command line."""

    task: str = ""
    cwd: Path | None = None
    max_turns: int | None = None
    resume: bool = False
    model: str | None = None  # None: default; PICK_MODEL: ask; else the ref

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
        return cls(task=" ".join(params.pop("task")).strip(), **params)

def _positive_int(_ctx: click.Context, _param: click.Parameter, value: int | None) -> int | None:
    if value is not None and value < 1:
        raise click.BadParameter(f"must be at least 1, got {value}")
    return value

def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))

def _force_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

@click.command(
    name=PROG,
    help="agent7 — minimal coding agent. Omit TASK to open the REPL.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", prog_name=PROG, message=f"{PROG} %(version)s")
@click.argument("task", nargs=-1)
@click.option("--cwd", type=click.Path(path_type=Path, file_okay=False),
              help="working directory (default: the current one)")
@click.option("--max-turns", type=int, callback=_positive_int, default=None, metavar="N",
              help=f"model rounds per task (default {DEFAULT_MAX_TURNS}, or $AGENT7_MAX_TURNS)")
@click.option("-m", "--model", "model", is_flag=False, flag_value=PICK_MODEL, default=None,
              metavar="REF",
              help="model for this run, e.g. deepseek-flash; a bare -m picks one from a menu")
@click.option("-r", "--resume", is_flag=True,
              help="resume the last session in this workspace")
def main(task: tuple[str, ...], cwd: Path | None, max_turns: int | None, resume: bool,
         model: str | None) -> None:
    _force_utf8_stdio()
    _load_dotenv()
    options = Options(task=" ".join(task).strip(), cwd=cwd, max_turns=max_turns, resume=resume,
                      model=model)
    try:
        asyncio.run(async_main(options))
    except KeyboardInterrupt:
        # Startup/cleanup happens outside the per-turn handler in repl_loop.
        err.print("\n[dim]# interrupted[/dim]")

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
    """fzp's Picker needs a terminal on both ends; a pipe cannot choose a model."""
    return bool(
        sys.stdin and sys.stdin.isatty() and sys.stderr and sys.stderr.isatty()
    )

def model_menu(session: Session, ids: list[str]) -> list[str]:
    """The refs a bare `-m` offers: the fetched list, else each provider's default."""
    if ids:
        return list(ids)
    return [
        f"{p.id}/{p.default_model}"
        for p in list_providers()
        if p.default_model and p.id in session.credentials
    ]

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

async def async_main(options: Options) -> None:
    """Run one task, or open the REPL, for an already-parsed command line."""
    task = options.task or None
    cwd = (options.cwd or Path.cwd()).resolve()
    saved = None
    try:
        saved = resume(cwd)
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
    workspace_model = os.environ.get("AGENT7_MODEL", "").strip() or (
        saved.get("model_ref", "") if saved else ""
    )
    model = workspace_model if options.model is None else options.model
    if pick and not can_pick_models():
        err.print("[red]# a bare -m shows a menu, which needs a terminal — "
                  "name the model instead: agent7 -m deepseek-flash[/red]")
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
        client = AsyncOpenAI(api_key="missing", base_url=list_providers()[0].default_base)
        if task:
            keys = ", ".join(k for p in list_providers() for k in p.env_keys)
            print(f"No credentials. Set {keys},\n"
                  "or run `uv run agent7` and `/login`.", file=sys.stderr)
            sys.exit(1)
        prov, cred = None, None
    state = Session(
        model=model or "pending", cwd=cwd, client=client, max_turns=max_turns,
        messages=resume_messages(snapshot, SYSTEM) if snapshot else [{"role": "system", "content": SYSTEM}],
        provider_id=prov.id if prov else None, cred_source=cred.source if cred else None,
        credential=cred, credentials=dict(all_creds),
    )
    # A one-shot becomes a real session only once it runs: a run that never
    # starts (bad -m, menu cancelled) must not replace the workspace snapshot.
    committed = False
    started = False

    def persist(commit: bool = False) -> None:
        # Merely opening a new REPL must not replace the snapshot used by -r.
        # The first agent turn makes it a real session and arms future saves.
        nonlocal committed
        committed = committed or commit
        if committed:
            save(state, created_at=created_at)

    try:
        ids: list[str] = []
        if all_creds:
            try:
                ids = await fetch_models(state)
            except Exception as e:  # noqa: BLE001
                err.print(f"[dim]# models fetch failed: {e}[/dim]")
        if pick:
            model = await choose_model(state, ids, current=workspace_model)
        if model:
            result = await switch_model(state, model)
            if result.message.startswith(("unknown", "ambiguous")):
                err.print(f"[dim]# {result.message}[/dim]")
                if options.model is not None:
                    # A ref the user typed must not quietly become another model.
                    raise SystemExit(1)
                if ids:
                    await switch_model(state, ids[0])
                else:
                    state.set_model(model)
        elif state.model in ("", "pending"):
            if ids:
                await switch_model(state, ids[0])
            else:
                state.set_model(default_model_for(state.provider_id))
        started = True
        emit_lifecycle(state, "session_started")
        err.print(f"[dim]# agent7 v{__version__}  model={state.model_ref or state.model}  "
                  f"providers=[{','.join(state.credentials) or 'none'}]  "
                  f"shell={SHELL.name}  cwd={state.cwd}[/dim]")
        action = "resumed" if snapshot else "new"
        err.print(f"[dim]# {action} session[/dim]")
        if snapshot:
            print_history(snapshot["messages"])
        if task:
            committed = True  # from here on the run owns the workspace snapshot
            try:
                await run_turn(state, task, on_saved=persist)
            except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
                err.print("\n[dim]# interrupted[/dim]")
            except Exception as e:  # noqa: BLE001
                err.print(f"[red]# error: {type(e).__name__}: {e}[/red]")
            return
        await repl_loop(state, persist)
    finally:
        if started:
            emit_lifecycle(state, "session_ended")
        try:
            persist()
        finally:
            await state.client.close()

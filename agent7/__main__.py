"""CLI: python -m agent7 [task] [--cwd DIR]"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding.vi_state import InputMode
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .agent import SHELL, SYSTEM, Interrupted, run_agent, tool_summary
from .commands import dispatch, fetch_models, known_names, switch_model
from .providers import (
    default_model_for,
    get_provider,
    list_providers,
    login_hints,
    resolve_all,
    resolve_default,
)
from .session import Session, format_context_prompt
from .store import resume, resume_messages, save, timestamp

console, err = Console(stderr=False), Console(stderr=True)
_THINK_HEADER = "  thinking"
_THINK_PREFIX = "  │ "
_THINK_ON, _THINK_OFF = "\033[36;2m", "\033[0m"

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

class ReplyPrinter:
    def __init__(self) -> None:
        self.buf: list[str] = []
        self._chars = 0
        self._thinking_open = False
        self._thinking_bol = True
        self._counting = False

    def __enter__(self) -> ReplyPrinter:
        return self

    def __exit__(self, *exc: object) -> None:
        self._close_thinking()
        self._clear_counter()
        text = "".join(self.buf)
        self.buf.clear()
        self._chars = 0
        if text.strip():
            console.print()
            console.print(Markdown(text))
            console.print()

    def _write(self, text: str) -> None:
        out = getattr(err, "file", None) or sys.stderr
        out.write(text)
        out.flush()

    def _paint(self, text: str) -> str:
        return f"{_THINK_ON}{text}{_THINK_OFF}" if err.is_terminal else text

    def _clear_status_line(self) -> None:
        if err.is_terminal:
            self._write("\r\033[K")

    def _close_thinking(self) -> None:
        if not self._thinking_open:
            return
        if not self._thinking_bol:
            self._write("\n")
        self._thinking_open, self._thinking_bol = False, True

    def _clear_counter(self) -> None:
        if self._counting:
            self._clear_status_line()
            self._counting = False

    def _show_counter(self) -> None:
        if not err.is_terminal:
            return
        self._close_thinking()
        self._counting = True
        self._write(f"\r\033[K  … {self._chars} chars")

    def _open_thinking(self) -> None:
        self._thinking_open, self._thinking_bol = True, True
        self._write("\n" + self._paint(_THINK_HEADER) + "\n")

    def on_thinking(self, text: str) -> None:
        """Stream the model's reasoning as dim text under a `thinking` header."""
        if not text:
            return
        self._clear_counter()
        if not self._thinking_open:
            self._open_thinking()
        lines = text.split("\n")
        chunks: list[str] = []
        for i, line in enumerate(lines):
            if line:
                if self._thinking_bol:
                    chunks.append(self._paint(_THINK_PREFIX))
                    self._thinking_bol = False
                chunks.append(self._paint(line))
            if i < len(lines) - 1:  # the stream continues on a new line
                chunks.append("\n")
                self._thinking_bol = True
        self._write("".join(chunks))

    def on_answer_to_thinking(self) -> None:
        # Content streamed before a tool call is not the final answer: drop it,
        # and end the thinking block so later reasoning opens a fresh one.
        self._close_thinking()
        if not self.buf:
            return
        self.buf.clear()
        self._chars = 0
        self._clear_counter()

    def on_delta(self, text: str) -> None:
        self.buf.append(text)
        self._chars += len(text)
        self._show_counter()

    def on_tool(self, turn: int, name: str, args: dict, result: str) -> None:
        self._close_thinking()
        self._clear_counter()
        self.buf.clear()
        self._chars = 0
        ok = not result.startswith("error:")
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        meta = ""
        if name == "bash" and result.startswith("exit="):
            code = result.split("\n", 1)[0].removeprefix("exit=")
            color = "green" if code == "0" else "yellow"
            meta = f"  [{color}]exit={code}[/{color}]"
        err.print(f"  {mark} [bold]{name:<5}[/bold] [dim]{tool_summary(name, args)}[/dim]{meta}")

    def on_preview(self, stats: str) -> None:
        self._close_thinking()
        self._clear_counter()
        line = (stats.splitlines() or [""])[0]
        if not line:
            return
        kind, sep, rest = line.partition("  ")
        if not sep or kind not in ("write", "edit"):
            kind, rest = "write", line
        path, _, counts = rest.rpartition("  ")
        head = Text(f"# {kind}  ", style="dim")
        head.append(path or rest, style="dim")
        if path and counts.startswith("+"):
            head.append("  ")
            for part in counts.split():
                head.append(part + " ", style="green bold" if part[0] == "+" else "red bold")
            head.right_crop(1)
        err.print(head)

_AT_PATH = re.compile(r"(?<![A-Za-z0-9_.\-/@])@((?:\\ |[^\s@])*)$")

def at_path_prefix(text: str) -> str | None:
    """The path fragment after an '@' that starts the current word ('@' -> '')."""
    m = _AT_PATH.search(text)
    return m.group(1) if m else None

def path_completions(cwd: Path, text: str) -> list[Completion]:
    """Complete files/dirs as `@path` for the word being typed at the end of text."""
    fragment = at_path_prefix(text)
    if fragment is None:
        return []
    raw = os.path.expanduser(fragment.replace("\\ ", " "))
    head, _, prefix = raw.rpartition("/")
    target = Path(head) if head else Path()
    if not target.is_absolute():
        target = cwd / target
    entries: list[tuple[Path, str]] = []
    show_hidden = prefix.startswith(".")
    for root, dirs, files in os.walk(target):
        if not show_hidden:
            dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in dirs + files:
            if (name.startswith(".") and not show_hidden) or not name.startswith(prefix):
                continue
            p = Path(root) / name
            entries.append((p, p.relative_to(target).as_posix()))
    entries.sort(key=lambda entry: (not entry[0].is_dir(), entry[1].lower()))
    out = []
    for p, path in entries:
        is_dir = p.is_dir()
        rel = f"{head}/{path}" if head else path
        out.append(Completion(
            (rel + "/" if is_dir else rel).replace(" ", "\\ "),
            start_position=-len(fragment),
            display=path + ("/" if is_dir else ""),
        ))
    return out

class ReplCompleter(Completer):
    """Slash commands, plus `@path` file completion from the cwd."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if not text.startswith("/"):
            yield from path_completions(self.session.cwd, text)
            return
        parts = text[1:].split(" ", 1)
        cmd = parts[0].lower()
        if len(parts) == 1 and " " not in text:
            for name in known_names():
                if name.startswith(cmd):
                    yield Completion(name, start_position=-len(cmd), display=f"/{name}")
            return
        if len(parts) == 2:
            prefix = parts[1]
            if cmd == "model":
                for mid in self.session.models_cache or []:
                    bare = mid.partition("/")[2]
                    if mid.startswith(prefix) or bare.startswith(prefix) or prefix.lower() in mid.lower():
                        yield Completion(mid, start_position=-len(prefix), display=mid)
            elif cmd in ("login", "logout"):
                names = [n for p in list_providers() for n in (p.id, *p.aliases)]
                for mid in names:
                    if mid.startswith(prefix.lower()):
                        yield Completion(mid, start_position=-len(prefix), display=mid)

NO_TEXT_NOTICE = "no text in the reply — the model stopped without content (retry, or /model)"
TRUNCATED_NOTICE = "no text in the reply — the model hit the output token limit (retry with a smaller step)"
STOPPED_PREFIX = "(stopped after "

def reply_notice(answer: str, finish_reason: str | None = None) -> str | None:
    """Dim note for a run that produced nothing visible on screen."""
    if answer.startswith(STOPPED_PREFIX):  # "ran out of turns" is itself the message
        return answer
    if answer.strip():
        return None
    return TRUNCATED_NOTICE if finish_reason == "length" else NO_TEXT_NOTICE

async def _run_once(session: Session, task: str, *, on_saved: Callable[[], None] | None = None) -> str:
    try:
        return await _run_agent_once(session, task)
    finally:
        if on_saved:
            on_saved()

async def _run_agent_once(session: Session, task: str) -> str:
    finish: list[str] = []
    with ReplyPrinter() as rp:
        answer = await run_agent(
            task, cwd=session.cwd, client=session.client, model=session.model,
            max_turns=session.max_turns, messages=session.messages,
            on_delta=rp.on_delta, on_thinking=rp.on_thinking,
            on_answer_to_thinking=rp.on_answer_to_thinking, on_tool=rp.on_tool,
            on_preview=rp.on_preview, on_usage=session.update_usage,
            context_limit=session.context_limit, prompt_tokens=session.prompt_tokens,
            on_compact=lambda n: err.print(f"[dim]# compacted {n} message(s)[/dim]"),
            on_finish=finish.append,
        )
    # never leave a tool call as the last thing on screen
    if notice := reply_notice(answer, finish[-1] if finish else None):
        err.print(f"[dim]# {notice}[/dim]")
    return answer

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="agent7 — minimal coding agent")
    p.add_argument("task", nargs="*", help="task; omit for REPL")
    p.add_argument("--cwd", type=Path, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("-r", "--resume", action="store_true",
                   help="resume the last session in this workspace")
    return p.parse_args(sys.argv[1:] if argv is None else argv)

async def async_main(argv: list[str] | None = None) -> None:
    _force_utf8_stdio()
    _load_dotenv()
    args = _parse_args(argv)
    task = " ".join(args.task).strip() or None
    cwd = (args.cwd or Path.cwd()).resolve()
    snapshot = None
    if args.resume:
        try:
            snapshot = resume(cwd)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    created_at = snapshot["created_at"] if snapshot else timestamp()
    max_turns = (
        args.max_turns if args.max_turns is not None
        else snapshot["max_turns"] if snapshot else 30
    )
    model = os.environ.get("AGENT7_MODEL") or (
        snapshot.get("model_ref", "") if snapshot else ""
    )

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
    def persist() -> None:
        save(state, created_at=created_at)
    try:
        ids: list[str] = []
        if all_creds:
            try:
                ids = await fetch_models(state)
            except Exception as e:  # noqa: BLE001
                err.print(f"[dim]# models fetch failed: {e}[/dim]")
        if model:
            result = await switch_model(state, model)
            if result.message.startswith(("unknown", "ambiguous")):
                err.print(f"[dim]# {result.message}[/dim]")
                if ids:
                    await switch_model(state, ids[0])
                else:
                    state.set_model(model)
        elif state.model in ("", "pending"):
            if ids:
                await switch_model(state, ids[0])
            else:
                state.set_model(default_model_for(state.provider_id))
        err.print(f"[dim]# agent7  model={state.model_ref or state.model}  "
                  f"providers=[{','.join(state.credentials) or 'none'}]  "
                  f"shell={SHELL.name}  cwd={state.cwd}[/dim]")
        action = "resumed" if snapshot else "new"
        err.print(f"[dim]# {action} session[/dim]")
        if task:
            try:
                await _run_once(state, task, on_saved=persist)
            except Interrupted:
                err.print("\n[dim]# interrupted[/dim]")
            return
        quit_hint = "Ctrl-Z to quit" if sys.platform == "win32" else "Ctrl-D to quit"
        err.print(f"[dim]# REPL  /help · /login · /model · @file · {quit_hint}[/dim]")
        if not all_creds:
            err.print(f"[dim]# no credentials — {login_hints()}[/dim]")

        def _vi_rprompt() -> HTML:
            mode = get_app().vi_state.input_mode
            label, color = ("INSERT", "ansigreen") if mode in (InputMode.INSERT, InputMode.INSERT_MULTIPLE) else (
                ("NORMAL", "ansiyellow") if mode == InputMode.NAVIGATION else ("REPLACE", "ansired")
            )
            return HTML(f'<style fg="{color}">-- {label} --</style>')

        prompt = PromptSession(
            history=InMemoryHistory(),
            completer=ReplCompleter(state),
            complete_while_typing=True,
            editing_mode=EditingMode.VI,
            rprompt=_vi_rprompt,
        )
        while True:
            try:
                line = (await prompt.prompt_async(lambda: f"{format_context_prompt(state)} > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not line:
                continue
            result = await dispatch(state, line)
            persist()
            if result is not None:
                for row in result.message.splitlines():
                    err.print(f"[dim]{row}[/dim]")
                continue
            if state.credential is None:
                err.print(f"[yellow]# login first: {login_hints()}[/yellow]")
                continue
            try:
                await _run_once(state, line, on_saved=persist)
            except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
                err.print("\n[dim]# interrupted[/dim]")
    finally:
        try:
            persist()
        finally:
            await state.client.close()

def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))

if __name__ == "__main__":
    main()

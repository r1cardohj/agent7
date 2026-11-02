"""CLI: python -m agent7 [task] [--cwd DIR]"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
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

from .agent import SYSTEM, Interrupted, run_agent, tool_summary
from .commands import dispatch, fetch_models, known_names, switch_model
from .providers import get_provider, resolve_all, resolve_default
from .session import Session, format_context_prompt

console, err = Console(stderr=False), Console(stderr=True)
_THINK_FRAMES = ("thinking   ", "thinking.  ", "thinking.. ", "thinking...")

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
        self._thinking_t0 = 0.0
        self._counting = False

    def __enter__(self) -> ReplyPrinter:
        return self

    def __exit__(self, *exc: object) -> None:
        self._close_thinking()
        self._clear_counter()
        text = "".join(self.buf)
        self.buf.clear()
        self._chars = 0
        if text:
            console.print()
            console.print(Markdown(text))
            console.print()

    def _clear_status_line(self) -> None:
        if err.is_terminal:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def _close_thinking(self) -> None:
        if self._thinking_open:
            self._clear_status_line()
            self._thinking_open = False

    def _clear_counter(self) -> None:
        if self._counting:
            self._clear_status_line()
            self._counting = False

    def _show_counter(self) -> None:
        if not err.is_terminal:
            return
        self._close_thinking()
        self._counting = True
        sys.stderr.write(f"\r\033[K  … {self._chars} chars")
        sys.stderr.flush()

    def on_thinking(self, text: str) -> None:
        if not text:
            return
        self._clear_counter()
        if not self._thinking_open:
            self._thinking_open, self._thinking_t0 = True, time.monotonic()
        if err.is_terminal:
            frame = int((time.monotonic() - self._thinking_t0) / 0.4) % len(_THINK_FRAMES)
            sys.stderr.write(f"\r\033[K  {_THINK_FRAMES[frame]}")
            sys.stderr.flush()

    def on_answer_to_thinking(self) -> None:
        if not self.buf:
            return
        self.buf.clear()
        self._chars = 0
        self._clear_counter()
        self.on_thinking(".")

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
        path, _, rest = line.rpartition("  ")
        head = Text("# write  ", style="dim")
        head.append(path or line, style="dim")
        if path and rest.startswith("+"):
            head.append("  ")
            for part in rest.split():
                head.append(part + " ", style="green bold" if part[0] == "+" else "red bold")
            head.right_crop(1)
        err.print(head)

class SlashCompleter(Completer):
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if not text.startswith("/"):
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
                for mid in ("deepseek", "kimi", "moonshot", "global"):
                    if mid.startswith(prefix.lower()):
                        yield Completion(mid, start_position=-len(prefix), display=mid)

async def _run_once(session: Session, task: str) -> str:
    with ReplyPrinter() as rp:
        return await run_agent(
            task, cwd=session.cwd, client=session.client, model=session.model,
            max_turns=session.max_turns, messages=session.messages,
            on_delta=rp.on_delta, on_thinking=rp.on_thinking,
            on_answer_to_thinking=rp.on_answer_to_thinking, on_tool=rp.on_tool,
            on_preview=rp.on_preview, on_usage=session.update_usage,
            context_limit=session.context_limit, prompt_tokens=session.prompt_tokens,
            on_compact=lambda n: err.print(f"[dim]# compacted {n} message(s)[/dim]"),
        )

async def async_main(argv: list[str] | None = None) -> None:
    _force_utf8_stdio()
    _load_dotenv()
    p = argparse.ArgumentParser(description="agent7 — minimal coding agent")
    p.add_argument("task", nargs="*", help="task; omit for REPL")
    p.add_argument("--cwd", type=Path, default=None)
    p.add_argument("--max-turns", type=int, default=30)
    args = p.parse_args(argv)
    task = " ".join(args.task).strip() or None
    cwd = (args.cwd or Path.cwd()).resolve()
    model = os.environ.get("AGENT7_MODEL") or ""

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
        client = AsyncOpenAI(api_key="missing", base_url="https://api.deepseek.com")
        if task:
            print("No credentials. Set DEEPSEEK_API_KEY / KIMI_API_KEY / MOONSHOT_API_KEY,\n"
                  "or run `uv run agent7` and `/login`.", file=sys.stderr)
            sys.exit(1)
        prov, cred = None, None
    state = Session(
        model=model or "pending", cwd=cwd, client=client, max_turns=args.max_turns,
        messages=[{"role": "system", "content": SYSTEM}],
        provider_id=prov.id if prov else None, cred_source=cred.source if cred else None,
        credential=cred, credentials=dict(all_creds),
    )
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
                state.set_model("deepseek-chat")
        err.print(f"[dim]# agent7  model={state.model_ref or state.model}  "
                  f"providers=[{','.join(state.credentials) or 'none'}]  cwd={state.cwd}[/dim]")
        if task:
            try:
                await _run_once(state, task)
            except Interrupted:
                err.print("\n[dim]# interrupted[/dim]")
            return
        err.print("[dim]# REPL  /help · /login · /model · empty line / Ctrl-D to quit[/dim]")
        if not all_creds:
            err.print("[dim]# no credentials — /login deepseek | /login kimi[/dim]")

        def _vi_rprompt() -> HTML:
            mode = get_app().vi_state.input_mode
            label, color = ("INSERT", "ansigreen") if mode in (InputMode.INSERT, InputMode.INSERT_MULTIPLE) else (
                ("NORMAL", "ansiyellow") if mode == InputMode.NAVIGATION else ("REPLACE", "ansired")
            )
            return HTML(f'<style fg="{color}">-- {label} --</style>')

        prompt = PromptSession(
            history=InMemoryHistory(),
            completer=SlashCompleter(state),
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
                break
            result = await dispatch(state, line)
            if result is not None:
                for row in result.message.splitlines():
                    err.print(f"[dim]{row}[/dim]")
                continue
            if state.credential is None:
                err.print("[yellow]# login first: /login deepseek | /login kimi[/yellow]")
                continue
            try:
                await _run_once(state, line)
            except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
                err.print("\n[dim]# interrupted[/dim]")
    finally:
        await state.client.close()

def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))

if __name__ == "__main__":
    main()

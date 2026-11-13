"""Terminal UI: reply rendering, prompt completion, and the REPL loop.

Everything here is about pixels and keystrokes.  `agent7.cli` parses the
command line and decides what to run; this module draws it.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path

from fzp.fuzzy import Match, fuzzy_filter, fuzzy_match
from fzp.tui import Picker
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app, get_app_session
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .agent import Interrupted, run_agent, tool_summary
from .commands import dispatch, known_names
from .hooks import emit_lifecycle
from .interrupt import run_with_interrupt_keys
from .mascot import print_mascot
from .pager import PromptToolkitPager
from .providers import list_providers, login_hints
from .routing import AUTO_REF
from .session import (
    Session,
    context_pressure,
    fmt_cwd,
    format_context_prompt,
)

console, err = Console(stderr=False), Console(stderr=True)


def _discard_paged_capture() -> None:
    """Close a capture level a failed pager left open.

    `PagerContext.__exit__` returns before `_exit_buffer()` when the pager
    raises, so the console would stay in "capture" mode and silently swallow
    every later reply. Rich offers no public way back out.
    """
    while getattr(console, "_buffer_index", 0) > 0:
        console._exit_buffer()
    buffer = getattr(console, "_buffer", None)
    if buffer is not None:
        buffer.clear()


_THINK_HEADER = "  thinking"
_THINK_PREFIX = "  │ "
_THINK_LINES = 6
_THINK_ON, _THINK_OFF = "\033[36;2m", "\033[0m"
_TOOL_TICK_INTERVAL = 1.0

class ReplyPrinter:
    def __init__(self) -> None:
        self.buf: list[str] = []
        self._chars = 0
        self._thinking_open = False
        self._thinking_bol = True
        self._thinking_text = ""
        self._thinking_screen_rows = 0
        self._counting = False
        self._waiting = False
        self._tool_running = False
        self._tool_started_at: float | None = None
        self._tool_serial = 0
        self._tool_status: Text | None = None
        self._tool_lock = threading.RLock()

    def __enter__(self) -> ReplyPrinter:
        return self

    def __exit__(self, *exc: object) -> None:
        self._close_thinking()
        self._clear_counter()
        self._clear_waiting()
        self._clear_tool()
        self._tool_started_at = None
        text = "".join(self.buf)
        self.buf.clear()
        self._chars = 0
        if text.strip():
            console.print()
            answer = Markdown(text)
            if console.is_terminal and sys.stdin.isatty():
                lines = console.render_lines(answer, pad=False)
                if len(lines) > console.height:
                    # pydoc falls back to more.com on Windows, which does not
                    # provide working arrow/page navigation in PowerShell.
                    pager = PromptToolkitPager() if sys.platform == "win32" else None
                    try:
                        with console.pager(pager=pager, styles=True):
                            console.print(answer)
                    except Exception:
                        # Never let a broken pager cost the user the answer;
                        # the print below is the fallback.
                        _discard_paged_capture()
            console.print(answer)
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
        if err.is_terminal:
            chunks = []
            if self._thinking_screen_rows > 1:
                chunks.append(f"\033[{self._thinking_screen_rows - 1}F")
            chunks.append("\r\033[J")
            self._write("".join(chunks))
        elif not self._thinking_bol:
            self._write("\n")
        self._thinking_open, self._thinking_bol = False, True
        self._thinking_text = ""
        self._thinking_screen_rows = 0

    def _clear_counter(self) -> None:
        if self._counting:
            self._clear_status_line()
            self._counting = False

    def _clear_waiting(self) -> None:
        if not self._waiting:
            return
        self._clear_status_line()
        self._waiting = False

    def _clear_tool_status(self) -> None:
        """Erase every terminal row occupied by the running tool status."""
        status = self._tool_status
        width = max(1, err.width)
        # Recalculate at erase time: a terminal resize may have reflowed the
        # status since it was drawn, so the original row count can be stale.
        rows = 1 if status is None else max(1, (status.cell_len + width - 1) // width)
        chunks = []
        if rows > 1:
            chunks.append(f"\033[{rows - 1}F")
        chunks.append("\r\033[J" if rows > 1 else "\r\033[K")
        self._write("".join(chunks))
        self._tool_status = None

    def _clear_tool(self) -> None:
        if not self._tool_running:
            return
        if err.is_terminal:
            self._clear_tool_status()
        else:
            self._write("\n")
        self._tool_running = False
        self._tool_status = None

    def on_model_start(self, turn: int) -> None:
        """Mark the otherwise silent gap before a model's first streamed delta."""
        self._clear_tool()
        self._close_thinking()
        self._clear_counter()
        if not err.is_terminal:
            return
        self._waiting = True
        self._write(f"\r\033[K  {self._paint('waiting...')}")

    def on_compact(self, count: int) -> None:
        self._clear_waiting()
        err.print(f"[dim]# compacted {count} message(s)[/dim]")

    def on_failover(self, old: str, new: str, error: BaseException) -> None:
        self._clear_waiting()
        reason = f"{type(error).__name__}: {error}".splitlines()[0]
        err.print(f"[yellow]# route: {old} → {new} ({reason[:180]})[/yellow]")

    def on_auto_selected(self, target: str, selection) -> None:
        self._clear_waiting()
        detail = selection.intent
        if (score := selection.scores.get(target)) is not None:
            detail += f" · {score:.0f}"
        if selection.error:
            detail += " · scoring unavailable"
        err.print(f"[dim]# auto: {target} · {detail}[/dim]")

    def _show_counter(self) -> None:
        self._clear_waiting()
        if not err.is_terminal:
            return
        self._close_thinking()
        self._counting = True
        self._write(f"\r\033[K  … {self._chars} chars")

    def _open_thinking(self) -> None:
        self._thinking_open, self._thinking_bol = True, True
        self._thinking_text = ""
        self._thinking_screen_rows = 0
        self._write("\n" + self._paint(_THINK_HEADER) + "\n")

    def _paint_thinking_viewport(self) -> None:
        rows: list[str] = []
        width = max(1, err.width - len(_THINK_PREFIX))
        for line in self._thinking_text.split("\n"):
            visual = Text(line)
            rows.extend(
                _THINK_PREFIX + part.plain if line else ""
                for part in visual.wrap(err, width)
            )
        visible = rows[-_THINK_LINES:]
        chunks = []
        if self._thinking_screen_rows > 1:
            chunks.append(f"\033[{self._thinking_screen_rows - 1}F")
        chunks.append("\r\033[J")
        for i, row in enumerate(visible):
            if i:
                chunks.append("\033[E")
            chunks.append(self._paint(row))
        self._thinking_screen_rows = len(visible)
        self._write("".join(chunks))

    def on_thinking(self, text: str) -> None:
        """Stream reasoning in a bounded terminal viewport, or in full to logs."""
        if not text:
            return
        self._clear_waiting()
        self._clear_counter()
        if not self._thinking_open:
            self._open_thinking()
        if err.is_terminal:
            self._thinking_text += text
            self._paint_thinking_viewport()
            return
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
        self._clear_waiting()
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

    def _draw_running_tool(
        self, name: str, args: dict, elapsed: int, serial: int,
    ) -> None:
        status = err.render_str(
            f"  [dim]•[/dim] [bold]{name:<5}[/bold] "
            f"[dim]{tool_summary(name, args)}  {elapsed}s[/dim]"
        )
        with self._tool_lock:
            if not self._tool_running or serial != self._tool_serial:
                return
            if err.is_terminal:
                self._clear_tool_status()
            err.print(status, end="", soft_wrap=True)
            if err.is_terminal:
                self._tool_status = status

    async def _tick_tool(self, name: str, args: dict, serial: int) -> None:
        shown = 0
        while self._tool_running and serial == self._tool_serial:
            await asyncio.sleep(_TOOL_TICK_INTERVAL)
            if not self._tool_running or serial != self._tool_serial:
                return
            started = self._tool_started_at
            if started is None:
                return
            elapsed = max(0, round(time.monotonic() - started))
            if elapsed != shown:
                shown = elapsed
                self._draw_running_tool(name, args, elapsed, serial)

    def on_tool_start(self, turn: int, name: str, args: dict) -> None:
        self._clear_waiting()
        self._close_thinking()
        self._clear_counter()
        self._clear_tool()
        self.buf.clear()
        self._chars = 0
        self._tool_running = True
        self._tool_serial += 1
        serial = self._tool_serial
        self._tool_started_at = time.monotonic()
        self._draw_running_tool(name, args, 0, serial)
        if err.is_terminal:
            try:
                # The serial check stops stale timers without cross-thread
                # Task.cancel(): on_preview runs in the tool worker thread.
                asyncio.get_running_loop().create_task(self._tick_tool(name, args, serial))
            except RuntimeError:
                pass  # Synchronous embedding still gets the initial 0s line.

    def on_tool(self, turn: int, name: str, args: dict, result: str) -> None:
        self._clear_waiting()
        self._clear_tool()
        self._close_thinking()
        self._clear_counter()
        self.buf.clear()
        self._chars = 0
        ok = not result.startswith("error:")
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        meta = ""
        if self._tool_started_at is not None:
            elapsed = max(0, round(time.monotonic() - self._tool_started_at))
            meta = f"  [dim]{elapsed}s[/dim]"
            self._tool_started_at = None
        if name == "bash" and result.startswith("exit="):
            code = result.split("\n", 1)[0].removeprefix("exit=")
            color = "green" if code == "0" else "yellow"
            meta += f"  [{color}]exit={code}[/{color}]"
        err.print(f"  {mark} [bold]{name:<5}[/bold] [dim]{tool_summary(name, args)}[/dim]{meta}")

    def on_preview(self, stats: str) -> None:
        # dispatch_tool invokes this from a worker thread; do not let a timer
        # repaint the running line in the middle of the preview.
        with self._tool_lock:
            self._print_preview(stats)

    def _print_preview(self, stats: str) -> None:
        self._clear_waiting()
        self._clear_tool()
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
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", ".next",
    ".gradle", ".idea", ".cache",
})
MAX_COMPLETIONS, MAX_SCANNED_ENTRIES = 300, 10_000
# A matched character is marked by color and weight only, never by a background:
# the background belongs to the row (see SELECTION_BG), so the hits of the
# selected row still read as one solid bar instead of green holes in it.
MATCH_STYLE = "fg:ansibrightgreen bold"
SELECTION_BG = "bg:#5f5f87"
PROMPT_STYLE = Style.from_dict({
    "bottom-toolbar": "noreverse bg:default",
    # prompt_toolkit paints the completion menu on grey unless told otherwise;
    # `fg:default bg:default` keeps it on the terminal's own colors.
    "completion-menu": "fg:default bg:default",
    "completion-menu.completion": "fg:default bg:default",
    # The selected row is the only thing that gets a background. `noreverse`
    # matters: with reverse on, the bright green of a match would turn into a
    # green block again.
    "completion-menu.completion.current": f"fg:#ffffff {SELECTION_BG} noreverse bold",
    "completion-menu.meta.completion": "fg:default bg:default",
    "completion-menu.meta.completion.current": f"fg:#c6c6d7 {SELECTION_BG}",
    "scrollbar.background": "bg:default",
    "scrollbar.button": "bg:default fg:ansibrightblack",
    "scrollbar.arrow": "bg:default fg:ansibrightblack",
})

def at_path_prefix(text: str) -> str | None:
    """The path fragment after an '@' that starts the current word ('@' -> '')."""
    m = _AT_PATH.search(text)
    return m.group(1) if m else None

def highlight(text: str, positions: tuple[int, ...]) -> list[tuple[str, str]]:
    """Formatted text that colors the characters a fuzzy match landed on."""
    if not positions:
        return [("", text)]
    parts: list[tuple[str, str]] = []
    last = 0
    for pos in positions:
        if pos > last:
            parts.append(("", text[last:pos]))
        parts.append((MATCH_STYLE, text[pos]))
        last = pos + 1
    if last < len(text):
        parts.append(("", text[last:]))
    return parts

def scan_paths(root: Path, *, recursive: bool, show_hidden: bool) -> Iterator[tuple[str, bool]]:
    """Path candidates below root as (path relative to root, is_dir).

    Bounded by MAX_SCANNED_ENTRIES so a huge tree can never stall the prompt;
    generated directories (.git, node_modules, …) are never entered.
    """
    stack: list[tuple[str, Path]] = [("", root)]
    seen = 0
    while stack and seen < MAX_SCANNED_ENTRIES:
        prefix, directory = stack.pop()
        try:
            with os.scandir(directory) as scanner:
                for entry in islice(scanner, MAX_SCANNED_ENTRIES - seen):
                    seen += 1
                    name = entry.name
                    if name.startswith(".") and not show_hidden:
                        continue
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        continue
                    if is_dir and name in SKIP_DIRS:
                        continue
                    yield prefix + name, is_dir
                    if recursive and is_dir:
                        stack.append((f"{prefix}{name}/", Path(entry.path)))
        except OSError:
            continue

def path_completions(cwd: Path, text: str) -> list[Completion]:
    """Fuzzy-complete files/dirs as `@path` for the word being typed at the end of text.

    `@` lists the cwd, `@dir/` lists that directory, and a bare fragment
    (`@sess`) searches the whole tree below the cwd, fzf-style.
    """
    fragment = at_path_prefix(text)
    if fragment is None:
        return []
    raw = os.path.expanduser(fragment.replace("\\ ", " "))
    head, _, prefix = raw.rpartition("/")
    target = Path(head).expanduser() if head else Path()
    if not target.is_absolute():
        target = cwd / target
    if not target.is_dir():
        return []
    # A directory hint, or a hidden-name prefix, keeps the listing local; a
    # bare name is a tree-wide fuzzy search.
    recursive = not head and bool(prefix) and not prefix.startswith(".")
    entries: list[tuple[str, bool, Match]] = []
    for path, is_dir in scan_paths(target, recursive=recursive, show_hidden=prefix.startswith(".")):
        match = fuzzy_match(prefix, path)
        if match is not None:
            entries.append((path, is_dir, match))
    # Score first, then directories, then lexical order for ties.
    entries.sort(key=lambda entry: (-entry[2].score, not entry[1], entry[0].lower()))
    out = []
    for path, is_dir, match in entries[:MAX_COMPLETIONS]:
        rel = f"{head}/{path}" if head else path
        display = path + ("/" if is_dir else "")
        out.append(Completion(
            (rel + "/" if is_dir else rel).replace(" ", "\\ "),
            start_position=-len(fragment),
            display=highlight(display, match.positions),
        ))
    return out

def repl_key_bindings() -> KeyBindings:
    """Enter submits; Ctrl-E edits the whole prompt in the external editor."""
    bindings = KeyBindings()
    default_focused = has_focus(DEFAULT_BUFFER)

    @bindings.add("enter", filter=default_focused)
    def _accept(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("c-e", filter=default_focused)
    def _edit(event: KeyPressEvent) -> None:
        # Do not validate_and_handle: :wq returns to the prompt for review.
        event.current_buffer.open_in_editor(validate_and_handle=False)

    return bindings

_STATUS_SEP = ("fg:ansibrightblack", " · ")

def vi_mode_label() -> tuple[str, str]:
    """The current vi input mode, and the color it is drawn in."""
    mode = get_app().vi_state.input_mode
    if mode in (InputMode.INSERT, InputMode.INSERT_MULTIPLE):
        return "INSERT", "ansigreen"
    if mode == InputMode.NAVIGATION:
        return "NORMAL", "ansiyellow"
    return "REPLACE", "ansired"

def context_tone(pct: int | None) -> str:
    """Green while there is room, yellow when it gets tight, red near the limit."""
    if pct is None or pct < 60:
        return "ansigreen"
    return "ansiyellow" if pct < 85 else "ansired"

def status_bar(
    state: Session,
    *,
    mode: tuple[str, str] | None = None,
    width: int | None = None,
) -> list[tuple[str, str]]:
    """One docked line under the prompt: model, context, cwd — mode on the right.

    Returned as prompt_toolkit formatted text so it can be rebuilt on every
    redraw: the context gauge and the vi mode change while you type.
    """
    if width is None:
        width = get_app().output.get_size().columns
    label, color = mode or vi_mode_label()
    pct = context_pressure(state)
    tone = context_tone(pct)
    segments = [("", " "), ("bold", state.display_model or "?")]
    segments += [_STATUS_SEP, (f"fg:{tone}", format_context_prompt(state))]
    if pct is not None:
        segments.append((f"fg:{tone}", f" {pct}%"))
    cwd = [_STATUS_SEP, ("fg:ansibrightblack", fmt_cwd(state.cwd))]
    right = [(f"fg:{color}", f" {label} ")]

    def used(parts: list[tuple[str, str]]) -> int:
        return sum(len(text) for _, text in parts)

    if used(segments + cwd + right) + 1 > width:
        cwd = []  # a narrow terminal keeps the model, the gauge, and the mode
    gap = max(1, width - used(segments + cwd + right))
    return segments + cwd + [("", " " * gap)] + right

class ReplCompleter(Completer):
    """Slash commands and `@path` files, both ranked by fzp's fuzzy score.

    prompt_toolkit shows completions in the order they are yielded and never
    filters them itself, so the fuzzy ranking (and the highlighted hits) has to
    happen here.  fzp's `fuzzy_filter` already returns the best match first.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if not text.startswith("/"):
            yield from path_completions(self.session.cwd, text)
            return
        raw, sep, prefix = text[1:].partition(" ")
        cmd = raw.lower()
        if not sep:
            for name, match in fuzzy_filter(cmd, known_names()):
                yield Completion(
                    name,
                    start_position=-len(raw),
                    display=highlight(f"/{name}", tuple(p + 1 for p in match.positions)),
                )
            return
        if cmd == "model":
            refs = list(self.session.models_cache or [])
            if getattr(self.session, "credentials", None):
                refs.append(AUTO_REF)
            for ref, match in fuzzy_filter(prefix, refs):
                yield Completion(ref, start_position=-len(prefix), display=highlight(ref, match.positions))
        elif cmd in ("login", "logout"):
            names = sorted({n for p in list_providers() for n in (p.id, *p.aliases)})
            for name, match in fuzzy_filter(prefix.lower(), names):
                yield Completion(name, start_position=-len(prefix), display=highlight(name, match.positions))

    async def get_completions_async(self, document: Document, complete_event: CompleteEvent):
        """Keep directory I/O off prompt_toolkit's event-loop thread."""
        text = document.text_before_cursor
        if not text.startswith("/"):
            completions = await asyncio.to_thread(path_completions, self.session.cwd, text)
            for completion in completions:
                yield completion
            return
        for completion in self.get_completions(document, complete_event):
            yield completion

def model_choices(models: list[str], current: str = "") -> list[str]:
    """Menu entries for a model pick: the current model first, then the rest.

    fzp's Picker deals in plain strings, so the model already in use is marked
    by position instead of a prefix that would leak into the chosen ref.
    """
    refs = [current, *models] if current else list(models)
    return list(dict.fromkeys(ref for ref in refs if ref))

async def pick_model(models: list[str], *, current: str = "") -> str | None:
    """Fuzzy-pick a model ref with fzp's Picker; None when the user aborts."""
    items = model_choices(models, current)
    if not items:
        return None
    # pick_async runs the picker on the caller's loop, so it fits inside the
    # one-shot coroutine; Ctrl-C / Esc resolve to None rather than raising.
    return await Picker(items).pick_async()


def session_choices(snapshots: list[dict], current: str = "") -> tuple[list[str], dict[str, str]]:
    """Readable fzp rows mapped back to their unambiguous session IDs."""
    ordered = sorted(snapshots, key=lambda data: data.get("id") != current)
    items, ids = [], {}
    for snapshot in ordered:
        session_id = snapshot.get("id", "")
        if not session_id:
            continue
        name = snapshot.get("name") or "New session"
        item = f"{name}  [{session_id}]"
        items.append(item)
        ids[item] = session_id
    return items, ids


async def pick_session(snapshots: list[dict], *, current: str = "") -> str | None:
    """Fuzzy-pick a session by its generated name; return its unique ID."""
    items, ids = session_choices(snapshots, current)
    if not items:
        return None
    picked = await Picker(items).pick_async()
    return ids.get(picked) if picked is not None else None


NO_TEXT_NOTICE = "no text in the reply — the model stopped without content (retry, or /model)"
TRUNCATED_NOTICE = "no text in the reply — the model hit the output token limit (retry with a smaller step)"
STOPPED_PREFIX = "(stopped after "
TURN_LIMIT_GUIDANCE = (
    "The task may be unfinished; your session is preserved.\n"
    "Type `continue` in the REPL, or run `agent7 -r \"continue\"`.\n"
    "To resume with a higher cap: `agent7 --max-turns 500 -r \"continue\"`."
)

def reply_notice(answer: str, finish_reason: str | None = None) -> str | None:
    """Explain an empty reply or a turn-limit pause, including how to recover."""
    if answer.startswith(STOPPED_PREFIX):
        return f"{answer}\n{TURN_LIMIT_GUIDANCE}"
    if answer.strip():
        return None
    return TRUNCATED_NOTICE if finish_reason == "length" else NO_TEXT_NOTICE

async def run_turn(session: Session, task: str, *, on_saved: Callable[[], None] | None = None) -> str:
    current = asyncio.current_task()
    previous_sigint = None
    if current and threading.current_thread() is threading.main_thread():
        # asyncio.run escalates its second SIGINT to KeyboardInterrupt, even when
        # the REPL handled the first one. Always cancel just this turn instead.
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: current.cancel())
    emit_lifecycle(session, "turn_started")
    try:
        work = _run_agent_once(session, task)
        if sys.stdin is not None and sys.stdin.isatty():
            answer = await run_with_interrupt_keys(work, input_obj=get_app_session().input)
        else:
            answer = await work
    except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
        emit_lifecycle(session, "turn_interrupted")
        raise
    except Exception as e:
        emit_lifecycle(session, "turn_failed", reason=type(e).__name__)
        raise
    else:
        if answer.startswith(STOPPED_PREFIX):
            emit_lifecycle(session, "turn_blocked", reason="max_turns")
        else:
            # Do not immediately replace done with idle just because the REPL
            # is about to prompt again: observers need the completion transition.
            emit_lifecycle(session, "turn_succeeded")
        return answer
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if on_saved:
            on_saved()

async def _run_agent_once(session: Session, task: str) -> str:
    finish: list[str] = []
    with ReplyPrinter() as rp:
        route_shown = False

        def on_route(target, previous, error) -> None:
            nonlocal route_shown
            session.use_route_target(target)
            if previous is not None:
                rp.on_failover(previous.ref, target.ref, error)
            elif not route_shown and session.route and session.route.selection:
                rp.on_auto_selected(target.ref, session.route.selection)
                route_shown = True

        if getattr(session, "route", None):
            rp.on_model_start(0)  # also cover the non-streaming scoring request
        answer = await run_agent(
            task, cwd=session.cwd, client=session.client, model=session.model,
            max_turns=session.max_turns, messages=session.messages,
            on_delta=rp.on_delta, on_thinking=rp.on_thinking,
            on_answer_to_thinking=rp.on_answer_to_thinking, on_tool=rp.on_tool,
            on_tool_start=rp.on_tool_start,
            on_preview=rp.on_preview, on_usage=session.update_usage,
            on_model_start=rp.on_model_start,
            context_limit=session.context_limit, prompt_tokens=session.prompt_tokens,
            on_compact=rp.on_compact,
            on_finish=finish.append,
            provider_id=getattr(session, "provider_id", None),
            route=getattr(session, "route", None), on_route=on_route,
        )
    # A turn-limit pause is a warning, not an invisible or successful final answer.
    if notice := reply_notice(answer, finish[-1] if finish else None):
        style = "yellow" if answer.startswith(STOPPED_PREFIX) else "dim"
        for line in notice.splitlines():
            err.print(f"[{style}]# {line}[/{style}]")
    return answer

def print_history(messages: list[dict]) -> None:
    """Replay the conversation text saved in a resumed session."""
    entries = [
        (message["role"], content)
        for message in messages
        if message.get("role") in ("user", "assistant")
        and isinstance(content := message.get("content"), str)
        and content.strip()
    ]
    if not entries:
        return
    console.print("[dim]# previous conversation[/dim]")
    for role, content in entries:
        console.print()
        console.print(f"[bold]{'you' if role == 'user' else 'assistant'}[/bold]")
        console.print(Markdown(content) if role == "assistant" else Text(content))
    console.print()

async def repl_loop(state: Session, persist: Callable[[bool], None]) -> None:
    """Read lines until Ctrl-D, running slash commands and agent turns.

    `persist(commit)` saves the session; `commit=True` also marks it as real,
    so opening a REPL and quitting untouched never overwrites the snapshot
    that `-r` would resume.
    """
    if err.is_terminal and not os.environ.get("AGENT7_NO_MASCOT"):
        print_mascot(err)
    quit_hint = "Ctrl-Z to quit" if sys.platform == "win32" else "Ctrl-D to quit"
    err.print(f"[dim]# REPL  /help · /session · /login · /model · @file · Ctrl-E editor · {quit_hint}[/dim]")
    if not state.credentials:
        err.print(f"[dim]# no credentials — {login_hints()}[/dim]")

    # Status and mode live in a bar docked under the prompt, so the prompt
    # itself stays a bare `> ` however much state the session carries.
    bar = None if os.environ.get("AGENT7_NO_STATUSBAR") else (lambda: status_bar(state))
    prompt = PromptSession(
        history=InMemoryHistory(),
        completer=ReplCompleter(state),
        complete_while_typing=True,
        editing_mode=EditingMode.VI,
        multiline=True,
        key_bindings=repl_key_bindings(),
        bottom_toolbar=bar,
        style=PROMPT_STYLE,
    )
    while True:
        try:
            line = (await prompt.prompt_async("> ")).strip()
        except EOFError:
            print(file=sys.stderr)
            if getattr(state, "persisted", False):
                err.print(f"[dim]# resume this session: agent7 -r {state.id}[/dim]")
            break
        except KeyboardInterrupt:
            # Ctrl-C abandons this input, not the whole REPL.
            print(file=sys.stderr)
            continue
        if not line:
            continue
        result = await dispatch(state, line)
        persist(False)
        if result is not None:
            for row in result.message.splitlines():
                err.print(f"[dim]{row}[/dim]")
            continue
        if state.credential is None:
            err.print(f"[yellow]# login first: {login_hints()}[/yellow]")
            continue
        persist(True)
        try:
            await run_turn(state, line, on_saved=lambda: persist(False))
        except (Interrupted, asyncio.CancelledError, KeyboardInterrupt):
            err.print("\n[dim]# interrupted[/dim]")
        except Exception as e:  # noqa: BLE001  a failed turn must not kill the REPL
            err.print(f"[red]# error: {type(e).__name__}: {e}[/red]")


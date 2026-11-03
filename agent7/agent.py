"""Async agent loop and tools: read, write, edit, bash (+ compact)."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

MAX_READ_CHARS, MAX_BASH_CHARS = 50_000, 12_000
DEFAULT_READ_LINES, MAX_TURNS = 200, 50
COMPACT_THRESHOLD, KEEP_USER_TURNS, KEEP_TOOL_RESULTS = 0.75, 3, 4
TOOL_MAX_CHARS, ASSISTANT_MAX_CHARS = 2_000, 4_000
DEEPSEEK_RL_SYSTEM = "You are a helpful software engineer assistant."

class Interrupted(Exception):
    def __init__(self, partial: str = "") -> None:
        self.partial = partial
        super().__init__(partial)

@dataclass(frozen=True)
class Shell:
    """Host shell used by the bash tool and named in the system prompt."""

    name: str  # bash | powershell | cmd
    argv: tuple[str, ...]  # prefix; the command string is appended as one arg

_SHELL_HINTS = {
    "bash": "rg, grep, find, ls (POSIX)",
    "powershell": "rg if available, Get-ChildItem, Select-String (PowerShell)",
    "cmd": "dir, findstr; rg if on PATH (cmd.exe)",
}

def _is_wsl_bash_stub(path: str) -> bool:
    # Windows ships System32\bash.exe as a WSL launcher — skip it for local Git Bash.
    norm = path.replace("/", "\\").lower()
    return norm.endswith("\\system32\\bash.exe") or norm.endswith("\\sysnative\\bash.exe")

def _git_bash_path() -> str | None:
    found = shutil.which("bash")
    if found and not _is_wsl_bash_stub(found):
        return found
    roots = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for root in roots:
        if not root:
            continue
        for rel in (r"Git\bin\bash.exe", r"Git\usr\bin\bash.exe", r"Programs\Git\bin\bash.exe"):
            cand = Path(root) / rel
            if cand.is_file():
                return str(cand)
    return None

def resolve_shell(*, platform: str | None = None) -> Shell:
    """Pick a shell from sys.platform (overridable via AGENT7_SHELL=bash|powershell|cmd|/path)."""
    plat = sys.platform if platform is None else platform
    override = os.environ.get("AGENT7_SHELL", "").strip()
    if override:
        key = override.lower()
        if key in ("bash", "powershell", "cmd", "sh"):
            return _shell_for_name("bash" if key == "sh" else key, plat)
        # Treat as executable path.
        base = Path(override).name.lower()
        if "bash" in base:
            return Shell("bash", (override, "-lc"))
        if base in ("pwsh", "pwsh.exe") or "powershell" in base:
            return Shell("powershell", (override, "-NoProfile", "-NonInteractive", "-Command"))
        return Shell("cmd", (override, "/c"))
    if plat == "win32":
        if bash := _git_bash_path():
            return Shell("bash", (bash, "-lc"))
        if pwsh := shutil.which("pwsh") or shutil.which("powershell"):
            return Shell("powershell", (pwsh, "-NoProfile", "-NonInteractive", "-Command"))
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return Shell("cmd", (comspec, "/c"))
    bash = shutil.which("bash") or "/bin/bash"
    return Shell("bash", (bash, "-lc"))

def _shell_for_name(name: str, platform: str) -> Shell:
    if name == "bash":
        if platform == "win32":
            bash = _git_bash_path() or shutil.which("bash") or "bash"
        else:
            bash = shutil.which("bash") or "/bin/bash"
        return Shell("bash", (bash, "-lc"))
    if name == "powershell":
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return Shell("powershell", (exe, "-NoProfile", "-NonInteractive", "-Command"))
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    return Shell("cmd", (comspec, "/c"))

def is_deepseek_model(model: str, *, provider_id: str | None = None) -> bool:
    """DeepSeek API models get the frozen RL persona; other providers keep the long prompt."""
    if (provider_id or "").lower() == "deepseek":
        return True
    return "deepseek" in (model or "").lower()

def make_system(
    shell: Shell | None = None,
    *,
    model: str | None = None,
    provider_id: str | None = None,
) -> str:
    if model is not None and is_deepseek_model(model, provider_id=provider_id):
        return DEEPSEEK_RL_SYSTEM
    sh = shell or resolve_shell()
    hint = _SHELL_HINTS.get(sh.name, sh.name)
    os_name = "Windows" if sys.platform == "win32" else ("macOS" if sys.platform == "darwin" else "Linux")
    return (
        "You are a coding agent. Tools: read, write, edit, bash. "
        "Explore before editing; read pages ~200 lines (offset/limit). "
        "Use read for file contents — do not cat/sed via bash. "
        f"Host OS: {os_name}. The bash tool runs commands via {sh.name} "
        "(cwd and env persist across calls). "
        f"Use bash for search and listing ({hint}). "
        "Use edit for precise changes (edits[].old_string must match exactly). "
        "When changing multiple places in one file, use one edit call with multiple edits[] "
        "instead of multiple edit calls. Each edits[].old_string is matched against the original "
        "file, not after earlier edits. Do not emit overlapping or nested edits; merge nearby "
        "changes into one edit. Keep old_string short but unique — do not pad with large "
        "unchanged regions. Use write only for new files or complete rewrites. "
        "Run tests only after a batch of edits. Finish with a short summary."
    )

def apply_system(messages: list[dict[str, Any]], system: str) -> None:
    """Replace or insert the leading system message in place."""
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system
    else:
        messages.insert(0, {"role": "system", "content": system})

SHELL = resolve_shell()
SYSTEM = make_system(SHELL)

def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }

_EDIT_ITEM = {
    "type": "object",
    "properties": {
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
    },
    "required": ["old_string", "new_string"],
}

TOOLS: list[dict[str, Any]] = [
    _fn("read", "Read text file (numbered). Default 200 lines; page with offset/limit.",
        {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["path"]),
    _fn("write", "Create or overwrite a whole file.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit", "Precise edits. One call may include multiple edits[] against the original file; no overlaps.",
        {"path": {"type": "string"}, "edits": {"type": "array", "items": _EDIT_ITEM}},
        ["path", "edits"]),
    _fn("bash", "Run a shell command (host shell). cwd and env persist across calls. Output capped ~12k chars.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
]

def _truncate(s: str, limit: int, *, tag: str = "truncated") -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[{tag} {len(s) - limit} chars]"

def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path.removeprefix("@").replace("\\ ", " "))  # tolerate `@path` typed in the REPL
    return p if p.is_absolute() else cwd / p

def tool_read(cwd: Path, path: str, offset: int | None = None, limit: int | None = None) -> str:
    fp = _resolve(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    try:
        lines = fp.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    total = len(lines)
    start = min(max(0, (offset or 1) - 1), total) if offset is not None else 0
    want = limit if limit is not None and limit > 0 else DEFAULT_READ_LINES
    end = min(total, start + want)
    body = "\n".join(f"{i + start + 1}|{line}" for i, line in enumerate(lines[start:end])) or "(empty)"
    body = _truncate(body, MAX_READ_CHARS)
    if start > 0 or end < total:
        hint = f"...[lines {start + 1}-{end} of {total}"
        if end < total:
            hint += f"; continue with offset={end + 1}"
        body = f"{body}\n{hint}]"
    return body

def _write_stats(fp: Path, before: str, after: str, *, kind: str = "write") -> str:
    a, b = before.splitlines(), after.splitlines()
    add = delete = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "insert":
            add += j2 - j1
        elif tag == "delete":
            delete += i2 - i1
        elif tag == "replace":
            delete += i2 - i1
            add += j2 - j1
    return f"{kind}  {fp}  +{add} -{delete}"

def tool_write(
    cwd: Path,
    path: str,
    content: str,
    on_preview: Callable[[str], None] | None = None,
) -> str:
    fp = _resolve(cwd, path)
    try:
        before = fp.read_text(encoding="utf-8") if fp.exists() else ""
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    after = content
    if before == after and fp.is_file():
        return f"no changes: {fp}"
    if on_preview:
        on_preview(_write_stats(fp, before, after, kind="write"))
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(after, encoding="utf-8", newline="\n")
    return f"wrote {len(after.encode('utf-8'))} bytes to {fp}"

def tool_edit(
    cwd: Path,
    path: str,
    edits: list[dict[str, Any]],
    on_preview: Callable[[str], None] | None = None,
) -> str:
    fp = _resolve(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    try:
        before = fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    if not isinstance(edits, list) or not edits:
        return "error: edits must be a non-empty array"
    spans: list[tuple[int, int, str, int]] = []
    for i, ed in enumerate(edits):
        if not isinstance(ed, dict):
            return f"error: edits[{i}] must be an object"
        old, new = ed.get("old_string"), ed.get("new_string")
        if not isinstance(old, str) or not old:
            return f"error: edits[{i}].old_string must be a non-empty string"
        if not isinstance(new, str):
            return f"error: edits[{i}].new_string must be a string"
        n = before.count(old)
        if n != 1:
            return f"error: edits[{i}].old_string occurs {n} times in {fp}; expected exactly 1"
        start = before.index(old)
        spans.append((start, start + len(old), new, i))
    ordered = sorted(spans, key=lambda s: s[0])
    for a, b in zip(ordered, ordered[1:]):
        if a[1] > b[0]:
            return (
                f"error: edits[{a[3]}] and edits[{b[3]}] overlap; "
                "merge nearby changes into one edit"
            )
    after = before
    for start, end, new, _ in sorted(spans, key=lambda s: s[0], reverse=True):
        after = after[:start] + new + after[end:]
    if before == after:
        return f"no changes: {fp}"
    if on_preview:
        on_preview(_write_stats(fp, before, after, kind="edit"))
    fp.write_text(after, encoding="utf-8", newline="\n")
    n = len(edits)
    return f"edited {n} span{'s' if n != 1 else ''} in {fp}"

def _bash_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    env["PS1"] = ""
    env["PROMPT_COMMAND"] = ""
    return env

def _bash_oneshot(cwd: Path, command: str, timeout: int | None = None) -> str:
    argv = [*SHELL.argv, command]
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout or 60, env=_bash_env(),
        )
    except FileNotFoundError:
        return f"error: shell not found: {SHELL.argv[0]}"
    except subprocess.TimeoutExpired as e:
        return _truncate(
            f"timeout after {timeout or 60}s\n{(e.stdout or '')}{(e.stderr or '')}", MAX_BASH_CHARS
        )
    parts = [f"exit={proc.returncode}"]
    if proc.stdout:
        parts.append("stdout:\n" + proc.stdout)
    if proc.stderr:
        parts.append("stderr:\n" + proc.stderr)
    return _truncate("\n".join(parts), MAX_BASH_CHARS)

class BashSession:
    """Persistent host shell: cwd and env survive across run() calls."""

    def __init__(self, shell: Shell | None = None, cwd: Path | None = None) -> None:
        self.shell = shell or SHELL
        self.cwd = (cwd or Path.cwd()).resolve()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._out_q: queue.Queue[bytes | None] = queue.Queue()

    def run(self, command: str, timeout: int | None = None) -> str:
        with self._lock:
            return self._run_locked(command, timeout or 60)

    def close(self) -> None:
        with self._lock:
            self._kill()

    def _persistent_argv(self) -> list[str]:
        exe = self.shell.argv[0]
        if self.shell.name == "bash":
            return [exe, "--noprofile", "--norc"]
        if self.shell.name == "powershell":
            return [exe, "-NoProfile", "-NonInteractive"]
        return [exe]  # cmd.exe reads commands from stdin

    def _wrap(self, command: str, token: str) -> str:
        if self.shell.name == "powershell":
            return (
                f"{command}\n"
                f"if ($null -eq $LASTEXITCODE) {{ $__e = 0 }} else {{ $__e = $LASTEXITCODE }}\n"
                f"Write-Output ('{token}:' + $__e)\n"
            )
        if self.shell.name == "cmd":
            return f"{command}\necho {token}:%ERRORLEVEL%\n"
        return f"{command}\necho {token}:$?\n"

    def _start(self) -> None:
        self._kill()
        try:
            self._proc = subprocess.Popen(
                self._persistent_argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=_bash_env(),
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(self.shell.argv[0]) from e
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            self._out_q.put(None)
            return
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    self._out_q.put(None)
                    return
                self._out_q.put(chunk)
        except Exception:  # noqa: BLE001
            self._out_q.put(None)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            with suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            with suppress(Exception):
                proc.terminate()
            with suppress(Exception):
                proc.wait(timeout=2)
            with suppress(Exception):
                proc.kill()
            with suppress(Exception):
                if proc.stdout:
                    proc.stdout.close()
        while True:
            try:
                self._out_q.get_nowait()
            except queue.Empty:
                break

    def _run_locked(self, command: str, timeout: int) -> str:
        try:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
        except FileNotFoundError:
            return f"error: shell not found: {self.shell.argv[0]}"
        assert self._proc and self._proc.stdin
        token = f"__AGENT7_{secrets.token_hex(16)}__"
        payload = self._wrap(command, token).encode("utf-8", errors="replace")
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            try:
                self._start()
            except FileNotFoundError:
                return f"error: shell not found: {self.shell.argv[0]}"
            assert self._proc and self._proc.stdin
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return f"error: shell write failed: {e}"
        deadline = time.monotonic() + timeout
        buf = bytearray()
        pattern = re.compile(re.escape(token) + r":(-?\d+)")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                text = buf.decode("utf-8", errors="replace")
                self._kill()
                return _truncate(f"timeout after {timeout}s\n{text}", MAX_BASH_CHARS)
            try:
                chunk = self._out_q.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
            if chunk is None:
                text = buf.decode("utf-8", errors="replace")
                self._kill()
                return _truncate(f"exit=?\nshell exited\n{text}", MAX_BASH_CHARS)
            buf.extend(chunk)
            text = buf.decode("utf-8", errors="replace")
            if m := pattern.search(text):
                body = text[: m.start()].rstrip("\n")
                parts = [f"exit={m.group(1)}"]
                if body:
                    parts.append("stdout:\n" + body)
                return _truncate("\n".join(parts), MAX_BASH_CHARS)

def tool_bash(
    cwd: Path,
    command: str,
    timeout: int | None = None,
    *,
    bash: BashSession | None = None,
) -> str:
    if bash is not None:
        return bash.run(command, timeout=timeout)
    return _bash_oneshot(cwd, command, timeout)

HANDLERS = {
    "read": tool_read, "write": tool_write, "edit": tool_edit, "bash": tool_bash,
}
_PREVIEW_TOOLS = frozenset({"write", "edit"})

def dispatch_tool(
    cwd: Path,
    name: str,
    args: dict[str, Any],
    *,
    on_preview: Callable[[str], None] | None = None,
    bash: BashSession | None = None,
) -> str:
    fn = HANDLERS.get(name)
    if not fn:
        return f"error: unknown tool {name}"
    try:
        if name in _PREVIEW_TOOLS:
            return fn(cwd, **args, on_preview=on_preview)
        if name == "bash":
            return fn(cwd, **args, bash=bash)
        return fn(cwd, **args)
    except TypeError as e:
        return f"error: bad args for {name}: {e}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"

def _trunc(s: str, n: int = 88) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"

def tool_summary(name: str, args: dict[str, Any]) -> str:
    if name == "bash":
        return _trunc(str(args.get("command") or ""))
    if name == "read":
        path = str(args.get("path") or "")
        off, lim = args.get("offset"), args.get("limit")
        return f"{path}  (offset={off} limit={lim})" if off is not None or lim is not None else path
    if name == "write":
        return f"{args.get('path', '')}  ({len(str(args.get('content') or ''))} chars)"
    if name == "edit":
        edits = args.get("edits")
        n = len(edits) if isinstance(edits, list) else 0
        return f"{args.get('path', '')}  ({n} edit{'s' if n != 1 else ''})"
    return _trunc(json.dumps(args, ensure_ascii=False))

def should_compact(
    prompt_tokens: int | None, context_limit: int | None, *, threshold: float = COMPACT_THRESHOLD
) -> bool:
    if prompt_tokens is None or context_limit is None or context_limit <= 0:
        return False
    return prompt_tokens >= int(context_limit * threshold)

def compact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_user_turns: int = KEEP_USER_TURNS,
    keep_tool_results: int = KEEP_TOOL_RESULTS,
    tool_max_chars: int = TOOL_MAX_CHARS,
    assistant_max_chars: int = ASSISTANT_MAX_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    if not messages:
        return messages, 0
    head, rest = ([], messages)
    if messages[0].get("role") == "system":
        head, rest = [messages[0]], messages[1:]
    user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    keep_from = user_idxs[-keep_user_turns] if len(user_idxs) >= keep_user_turns else 0
    tool_idxs = [i for i, m in enumerate(rest) if m.get("role") == "tool"]
    old_tool_before = keep_from or len(rest)
    if keep_from == 0 and tool_idxs:
        old_tool_before = (
            len(rest) if keep_tool_results <= 0
            else tool_idxs[max(0, len(tool_idxs) - keep_tool_results)]
        )
    out, changed = list(head), 0
    for i, m in enumerate(rest):
        role, content = m.get("role"), m.get("content")
        if role == "tool" and i < old_tool_before and isinstance(content, str) and len(content) > tool_max_chars:
            out.append({**m, "content": _truncate(content, tool_max_chars, tag="compacted")})
            changed += 1
        elif (
            role == "assistant" and i < keep_from and isinstance(content, str)
            and len(content) > assistant_max_chars and not m.get("tool_calls")
        ):
            out.append({**m, "content": _truncate(content, assistant_max_chars, tag="compacted")})
            changed += 1
        else:
            out.append(m)
    return out, changed

async def _emit(cb: Callable[..., Any] | None, *args: Any) -> None:
    if cb and asyncio.iscoroutine(maybe := cb(*args)):
        await maybe

def _delta_reasoning(delta: Any) -> str | None:
    if text := getattr(delta, "reasoning_content", None):
        return text
    extra = getattr(delta, "model_extra", None) or {}
    return str(extra["reasoning_content"]) if isinstance(extra, dict) and extra.get("reasoning_content") else None

def _usage_from_chunk(chunk: Any) -> dict[str, int] | None:
    usage = getattr(chunk, "usage", None)
    if not usage:
        return None
    out = {k: int(v) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
           if (v := getattr(usage, k, None)) is not None}
    return out or None

async def _stream_completion(
    client: AsyncOpenAI, *, model: str, messages: list[dict[str, Any]],
    on_delta, on_thinking, on_answer_to_thinking, on_finish=None,
) -> tuple[str, list[dict[str, Any]], dict[str, int] | None]:
    kwargs: dict[str, Any] = {
        "model": model, "messages": messages, "tools": TOOLS, "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        stream = await client.chat.completions.create(**kwargs)
    except Interrupted:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted() from e
    except Exception:
        kwargs.pop("stream_options", None)
        try:
            stream = await client.chat.completions.create(**kwargs)
        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            raise Interrupted() from e
    parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    saw_tools, usage = False, None
    try:
        async for chunk in stream:
            if u := _usage_from_chunk(chunk):
                usage = u
            if not chunk.choices:
                continue
            if finish := getattr(chunk.choices[0], "finish_reason", None):
                await _emit(on_finish, finish)  # "stop" | "length" | "tool_calls" | ...
            delta = chunk.choices[0].delta
            if r := _delta_reasoning(delta):
                await _emit(on_thinking, r)
            if delta.tool_calls:
                if not saw_tools:
                    saw_tools = True
                    await _emit(on_answer_to_thinking)
                for tc in delta.tool_calls:
                    slot = tool_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            if delta.content:
                parts.append(delta.content)
                await _emit(on_thinking if saw_tools else on_delta, delta.content)
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted("".join(parts)) from e
    finally:
        if close := getattr(stream, "close", None):
            with suppress(Exception):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
    return "".join(parts), [
        {"id": tool_acc[i]["id"], "type": "function",
         "function": {"name": tool_acc[i]["name"], "arguments": tool_acc[i]["arguments"] or "{}"}}
        for i in sorted(tool_acc)
    ], usage

async def run_agent(
    task: str, *, cwd: Path, client: AsyncOpenAI, model: str, max_turns: int = MAX_TURNS,
    messages: list[dict[str, Any]], on_delta=None, on_thinking=None, on_answer_to_thinking=None,
    on_tool=None, on_preview=None, on_usage=None, context_limit: int | None = None,
    prompt_tokens: int | None = None, on_compact=None, on_finish=None,
    bash: BashSession | None = None,
) -> str:
    cwd = cwd.resolve()
    messages.append({"role": "user", "content": task})
    last_prompt = prompt_tokens
    own_bash = bash is None
    if own_bash:
        bash = BashSession(SHELL, cwd)
    try:
        for turn in range(1, max_turns + 1):
            if should_compact(last_prompt, context_limit):
                new_msgs, n = compact_messages(messages)
                if n:
                    messages[:] = new_msgs
                    await _emit(on_compact, n)
            try:
                content, tool_calls, usage = await _stream_completion(
                    client, model=model, messages=messages, on_delta=on_delta,
                    on_thinking=on_thinking, on_answer_to_thinking=on_answer_to_thinking,
                    on_finish=on_finish,
                )
            except Interrupted as e:
                if e.partial:
                    messages.append({"role": "assistant", "content": e.partial})
                raise
            if usage:
                await _emit(on_usage, usage)
                last_prompt = usage.get("prompt_tokens", last_prompt)
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            if not tool_calls:
                return content
            for tc in tool_calls:
                name, raw_args = tc["function"]["name"], tc["function"]["arguments"]
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args, result = {}, f"error: invalid JSON arguments: {raw_args!r}"
                else:
                    try:
                        result = await asyncio.to_thread(
                            dispatch_tool, cwd, name, args, on_preview=on_preview, bash=bash
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt) as e:
                        messages.append(
                            {"role": "tool", "tool_call_id": tc["id"], "content": "error: interrupted"}
                        )
                        raise Interrupted() from e
                if on_tool:
                    await _emit(on_tool, turn, name, args, result)
                else:
                    print(f"  [{turn}] {name} {tool_summary(name, args)}", file=sys.stderr, flush=True)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return f"(stopped after {max_turns} turns)"
    except Interrupted:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted() from e
    finally:
        if own_bash and bash is not None:
            bash.close()

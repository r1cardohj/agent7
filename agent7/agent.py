"""Async agent loop and tools: read, write, edit, bash (+ compact)."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .responses import (
    RAW_ITEMS_KEY,
    ResponseTurn,
    discard_raw_output,
    output_text,
    plain_item,
    portable_history,
    public_call,
    raw_state,
    reasoning_text,
    request_input,
    response_status,
    response_tools,
    response_usage,
)

MAX_READ_CHARS, MAX_BASH_CHARS = 50_000, 12_000
DEFAULT_READ_LINES, DEFAULT_MAX_TURNS = 200, 200  # turns are a safety net, not a budget
COMPACT_THRESHOLD, KEEP_USER_TURNS, KEEP_TOOL_RESULTS = 0.75, 3, 4
TOOL_MAX_CHARS, ASSISTANT_MAX_CHARS = 2_000, 4_000
OVERFLOW_RETRIES = 2
RETRY_SLEEPS = (0.5, 1.5, 3.0)
STREAM_CLOSE_TIMEOUT = 2.0
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
OVERFLOW_STATUS = frozenset({400, 413, 422})

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

def make_system(shell: Shell | None = None) -> str:
    sh = shell or resolve_shell()
    hint = _SHELL_HINTS.get(sh.name, sh.name)
    os_name = "Windows" if sys.platform == "win32" else ("macOS" if sys.platform == "darwin" else "Linux")
    return (
        "You are a coding agent. Tools: read, write, edit, bash. "
        "Explore before editing; read pages ~200 lines (offset/limit). "
        "Use read for file contents — do not cat/sed via bash. "
        f"Host OS: {os_name}. The bash tool runs commands via {sh.name}. "
        f"Use bash for search and listing ({hint}). "
        "Use edit for precise changes (edits[].old_string must match exactly). "
        "When changing multiple places in one file, use one edit call with multiple edits[] "
        "instead of multiple edit calls. Each edits[].old_string is matched against the original "
        "file, not after earlier edits. Do not emit overlapping or nested edits; merge nearby "
        "changes into one edit. Keep old_string short but unique — do not pad with large "
        "unchanged regions. Use write only for new files or complete rewrites. "
        "Run tests only after a batch of edits. "
        "When creating a git commit, add a blank line followed by "
        "`Co-authored-by: agent7-bot <160873036+agent7-bot@users.noreply.github.com>` "
        "unless the user asks otherwise. Finish with a short summary."
    )

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
    _fn("bash", "Run a shell command in cwd (host shell). Output capped ~12k chars.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
]

def _truncate(s: str, limit: int, *, tag: str = "truncated") -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[{tag} {len(s) - limit} chars]"

def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path.removeprefix("@").replace("\\ ", " ")).expanduser()  # `@path` from the REPL
    return p if p.is_absolute() else cwd / p

def _write_atomic(fp: Path, text: str) -> None:
    """Temp file + os.replace: a Ctrl-C mid-write can never truncate the file."""
    target = fp.resolve() if fp.is_symlink() else fp  # keep symlinks pointing where they did
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        with suppress(OSError):
            os.chmod(tmp, target.stat().st_mode & 0o7777)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

def _count_lines(fp: Path) -> int:
    """Total line count at C speed (bytes.count) so a 100 MB log stays fast."""
    total, tail = 0, b"\n"
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            total += chunk.count(b"\n")
            tail = chunk[-1:]
    return total if tail == b"\n" or total == 0 else total + 1

def tool_read(cwd: Path, path: str, offset: int | None = None, limit: int | None = None) -> str:
    fp = _resolve(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    want = limit if limit is not None and limit > 0 else DEFAULT_READ_LINES
    start = max(0, (offset or 1) - 1) if offset is not None else 0
    total = _count_lines(fp)
    window: list[str] = []
    kept = end = 0
    try:
        # Only the wanted window is decoded and kept, and no single line can be
        # unbounded: a 100 MB log costs a few hundred KB instead of 300 MB.
        with fp.open(encoding="utf-8") as fh:
            lineno = 0
            while line := fh.readline(MAX_READ_CHARS + 1):
                lineno += 1
                if lineno > start + want or kept >= MAX_READ_CHARS:
                    break
                if lineno <= start:
                    continue
                text = line.rstrip("\r\n")
                window.append(text)
                kept += len(text) + 1
                end = lineno
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    end = end or min(start, total)
    body = "\n".join(f"{i + start + 1}|{line}" for i, line in enumerate(window)) or "(empty)"
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
    _write_atomic(fp, after)
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
    _write_atomic(fp, after)
    n = len(edits)
    return f"edited {n} span{'s' if n != 1 else ''} in {fp}"

_BASH_HEAD_BYTES = MAX_BASH_CHARS * 4  # one char is at most 4 bytes in UTF-8

def _head(f: Any) -> str:
    """First ~MAX_BASH_CHARS chars of a temp file; the rest never enters memory."""
    f.flush()
    f.seek(0)
    return f.read(_BASH_HEAD_BYTES).decode("utf-8", "replace")

def _process_group_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if proc.poll() is None:
        proc.kill()


def tool_bash(cwd: Path, command: str, timeout: int | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    argv = [*SHELL.argv, command]
    limit = timeout or 60
    # Output goes to temp files instead of buffering everything in memory. The
    # shell gets its own process group so a timeout kills its children as well.
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, stdout=out_f, stderr=err_f, env=env,
                **_process_group_options(),
            )
        except FileNotFoundError:
            return f"error: shell not found: {SHELL.argv[0]}"
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            proc.wait()
            return _truncate(
                f"timeout after {limit}s\n{_head(out_f)}{_head(err_f)}", MAX_BASH_CHARS
            )
        parts = [f"exit={proc.returncode}"]
        if stdout := _head(out_f):
            parts.append("stdout:\n" + stdout)
        if stderr := _head(err_f):
            parts.append("stderr:\n" + stderr)
    return _truncate("\n".join(parts), MAX_BASH_CHARS)

HANDLERS = {
    "read": tool_read, "write": tool_write, "edit": tool_edit, "bash": tool_bash,
}
_PREVIEW_TOOLS = frozenset({"write", "edit"})

def dispatch_tool(
    cwd: Path, name: str, args: dict[str, Any], *, on_preview: Callable[[str], None] | None = None
) -> str:
    fn = HANDLERS.get(name)
    if not fn:
        return f"error: unknown tool {name}"
    try:
        return fn(cwd, **args, on_preview=on_preview) if name in _PREVIEW_TOOLS else fn(cwd, **args)
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


class _Retryable(Exception):
    """Failed before any text reached the screen, so the request can be replayed."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(str(cause))

_RETRY_ERRORS = frozenset({
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "ServiceUnavailableError", "BadGatewayError",
})

def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in RETRY_STATUS
    return type(exc).__name__ in _RETRY_ERRORS


def _is_context_overflow(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) not in OVERFLOW_STATUS:
        return False
    text = f"{exc} {getattr(exc, 'body', '')}".lower()
    return any(k in text for k in (
        "context_length", "context length", "maximum context", "too many tokens",
        "prompt is too long", "input is too long",
    ))

def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough size for providers that never report usage: ~4 chars per token."""
    chars = 0
    for m in messages:
        chars += len(str(m.get("content") or ""))
        for call in m.get("tool_calls") or []:
            chars += len(str((call.get("function") or {}).get("arguments") or ""))
    return chars // 4 + 1

async def _open_stream(client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
    try:
        return await client.responses.create(**kwargs)
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted() from e

def _consume_close_result(task: asyncio.Task) -> None:
    """Retrieve a detached close task's exception after an interrupted cleanup."""
    if not task.cancelled():
        task.exception()

async def _stream_once(
    client: AsyncOpenAI, kwargs: dict[str, Any], *, on_delta, on_thinking,
    on_answer_to_thinking, on_finish,
) -> ResponseTurn:
    try:
        stream = await _open_stream(client, kwargs)
    except Interrupted:
        raise
    except Exception as e:
        raise _Retryable(e) from e
    parts: list[str] = []
    output: list[dict[str, Any]] = []
    saw_tools = emitted = False
    response = None
    try:
        async for event in stream:
            kind = getattr(event, "type", "")
            if kind == "response.output_item.done":
                item = plain_item(event.item)
                output.append(item)
                if item.get("type") == "function_call":
                    if not saw_tools:
                        saw_tools = True
                        await _emit(on_answer_to_thinking)
                elif item.get("type") == "message":
                    text = output_text([item])
                    if text and text != "".join(parts):
                        # Some compatible endpoints return a final item but no deltas.
                        emitted = True
                        parts.append(text)
                        await _emit(on_thinking if saw_tools else on_delta, text)
                elif item.get("type") == "reasoning" and not emitted:
                    # Delta-less compatible endpoints still expose useful summaries.
                    if text := reasoning_text(item):
                        emitted = True
                        await _emit(on_thinking, text)
            elif kind in (
                "response.reasoning_text.delta", "response.reasoning_summary_text.delta",
            ):
                if text := getattr(event, "delta", ""):
                    emitted = True
                    await _emit(on_thinking, text)
            elif kind == "response.output_text.delta":
                if text := getattr(event, "delta", ""):
                    emitted = True
                    parts.append(text)
                    await _emit(on_thinking if saw_tools else on_delta, text)
            elif kind in ("response.completed", "response.incomplete"):
                response = event.response
                final_output = [plain_item(item) for item in getattr(response, "output", None) or []]
                if final_output:
                    output = final_output
                    if not parts and (text := output_text(output)):
                        parts.append(text)
                        emitted = True
                        await _emit(on_thinking if saw_tools else on_delta, text)
                finish, _ = response_status(response)
                await _emit(on_finish, finish)
            elif kind == "response.failed":
                response = event.response
                error = getattr(response, "error", None)
                raise RuntimeError(getattr(error, "message", None) or "Responses API request failed")
            elif kind == "response.error":
                error = getattr(event, "error", None)
                raise RuntimeError(getattr(error, "message", None) or "Responses API stream failed")
    except Interrupted:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted("".join(parts)) from e
    except Exception as e:
        if emitted:  # the user already saw text: replaying it would duplicate it
            raise
        raise _Retryable(e) from e
    finally:
        if close := getattr(stream, "close", None):
            with suppress(Exception):
                result = close()
                if asyncio.iscoroutine(result):
                    interrupted = isinstance(sys.exc_info()[1], Interrupted)
                    # Shield the close so its timeout cannot wait for cleanup that
                    # ignores cancellation. Detached tasks still report their errors.
                    closing = asyncio.create_task(result)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(closing), timeout=STREAM_CLOSE_TIMEOUT
                        )
                    except BaseException:
                        closing.cancel()
                        closing.add_done_callback(_consume_close_result)
                        if not interrupted:
                            raise
                        # Preserve the original interrupt over cleanup cancellation.
    finish = response_status(response)[0] if response is not None else None
    return ResponseTurn(
        text="".join(parts), output=output, usage=response_usage(response),
        finish_reason=finish,
    )

async def _stream_completion(
    client: AsyncOpenAI, *, model: str, messages: list[dict[str, Any]],
    on_delta, on_thinking, on_answer_to_thinking, on_finish=None,
    provider_id: str | None = None,
) -> ResponseTurn:
    inputs, instructions = request_input(messages, provider_id)
    kwargs: dict[str, Any] = {
        "model": model, "input": inputs, "tools": response_tools(TOOLS), "stream": True,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if provider_id == "deepseek" and portable_history(messages, provider_id):
        # DeepSeek rejects thinking-mode replay when legacy/other-provider
        # history has no reasoning items. Key off the provider, not the model
        # name. Degrade only until fresh raw output is available again.
        kwargs["reasoning"] = {"effort": "none"}
    for attempt in range(len(RETRY_SLEEPS) + 1):
        try:
            return await _stream_once(
                client, kwargs, on_delta=on_delta, on_thinking=on_thinking,
                on_answer_to_thinking=on_answer_to_thinking, on_finish=on_finish,
            )
        except _Retryable as e:
            if attempt == len(RETRY_SLEEPS) or not _retryable(e.cause):
                raise e.cause from None
            await asyncio.sleep(RETRY_SLEEPS[attempt])
    raise AssertionError("unreachable")  # pragma: no cover

def _answer_pending(
    messages: list[dict[str, Any]], calls: list[dict[str, Any]],
    note: str = "error: interrupted",
) -> None:
    """Answer every tool call that will never run, so the history stays replayable."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for call in calls:
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id and call_id not in answered:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": note})

async def run_agent(
    task: str, *, cwd: Path, client: AsyncOpenAI, model: str, max_turns: int = DEFAULT_MAX_TURNS,
    messages: list[dict[str, Any]], on_delta=None, on_thinking=None, on_answer_to_thinking=None,
    on_tool=None, on_tool_start=None, on_preview=None, on_usage=None, context_limit: int | None = None,
    prompt_tokens: int | None = None, on_compact=None, on_finish=None, on_model_start=None,
    provider_id: str | None = None,
) -> str:
    cwd = cwd.resolve()
    messages.append({"role": "user", "content": task})
    last_prompt = prompt_tokens
    turns = retries = 0
    try:
        while turns < max_turns:
            # Providers that never report usage would otherwise never compact.
            prompt = last_prompt if last_prompt is not None else estimate_tokens(messages)
            if should_compact(prompt, context_limit):
                new_msgs, n = compact_messages(messages)
                if n:
                    discard_raw_output(new_msgs)
                    messages[:] = new_msgs
                    last_prompt = None
                    await _emit(on_compact, n)
            # The gap after a tool result is a new model wait, not a running tool.
            await _emit(on_model_start, turns + 1)
            try:
                turn = await _stream_completion(
                    client, model=model, messages=messages, on_delta=on_delta,
                    on_thinking=on_thinking, on_answer_to_thinking=on_answer_to_thinking,
                    on_finish=on_finish, provider_id=provider_id,
                )
                content = turn.text
                tool_calls = [public_call(call) for call in turn.calls]
                usage = turn.usage
            except Interrupted as e:
                if e.partial:
                    messages.append({"role": "assistant", "content": e.partial})
                raise
            except Exception as e:
                # Server-side context overflow: shrink hard, then retry the same turn.
                # The retry makes no progress, so it must not spend the turn budget.
                retries += 1
                squeezed: list[dict[str, Any]] = []
                n = 0
                if retries <= OVERFLOW_RETRIES and _is_context_overflow(e):
                    squeezed, n = compact_messages(
                        messages, keep_user_turns=1, keep_tool_results=1
                    )
                if not n:
                    raise
                discard_raw_output(squeezed)
                messages[:] = squeezed
                last_prompt = None
                await _emit(on_compact, n)
                continue
            retries = 0
            turns += 1
            if usage:
                await _emit(on_usage, usage)
                last_prompt = usage.get("prompt_tokens", last_prompt)
            assistant: dict[str, Any] = {
                "role": "assistant", "content": content,
                RAW_ITEMS_KEY: raw_state(provider_id, turn.output),
            }
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
                    await _emit(on_tool_start, turns, name, args)
                    try:
                        result = await asyncio.to_thread(
                            dispatch_tool, cwd, name, args, on_preview=on_preview
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt) as e:
                        # Every tool_call needs a tool message, or the next request is a 400.
                        _answer_pending(messages, tool_calls)
                        raise Interrupted() from e
                if on_tool:
                    await _emit(on_tool, turns, name, args, result)
                else:
                    print(f"  [{turns}] {name} {tool_summary(name, args)}", file=sys.stderr, flush=True)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return f"(stopped after {max_turns} turns)"
    except Interrupted:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise Interrupted() from e

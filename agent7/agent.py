"""Async agent loop and tools: read, write, bash, grep (+ compact)."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

MAX_READ_CHARS, MAX_BASH_CHARS, MAX_GREP_CHARS = 50_000, 12_000, 12_000
DEFAULT_READ_LINES, MAX_GREP_LINES, MAX_TURNS = 200, 100, 30
COMPACT_THRESHOLD, KEEP_USER_TURNS, KEEP_TOOL_RESULTS = 0.75, 3, 4
TOOL_MAX_CHARS, ASSISTANT_MAX_CHARS = 2_000, 4_000
class Interrupted(Exception):
    def __init__(self, partial: str = "") -> None:
        self.partial = partial
        super().__init__(partial)

SYSTEM = (
    "You are a coding agent. Tools: read, write, bash, grep. "
    "Explore before editing; read pages ~200 lines (offset/limit). "
    "Prefer write(old_string=...) for small edits. Finish with a short summary."
)

def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }

TOOLS: list[dict[str, Any]] = [
    _fn("read", "Read text file (numbered). Default 200 lines; page with offset/limit.",
        {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["path"]),
    _fn("write", "Write file, or replace one unique span via old_string.",
        {"path": {"type": "string"}, "content": {"type": "string"},
         "old_string": {"type": "string"}}, ["path", "content"]),
    _fn("bash", "Shell in cwd. Output capped ~12k chars.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
    _fn("grep", "rg/grep search. Capped at 100 lines.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}},
        ["pattern"]),
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

def _write_stats(fp: Path, before: str, after: str) -> str:
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
    return f"{fp}  +{add} -{delete}"

def tool_write(
    cwd: Path,
    path: str,
    content: str,
    old_string: str | None = None,
    on_preview: Callable[[str], None] | None = None,
) -> str:
    fp = _resolve(cwd, path)
    try:
        before = fp.read_text(encoding="utf-8") if fp.exists() else ""
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    if old_string is not None:
        if not fp.is_file():
            return f"error: not a file: {fp}"
        if not old_string:
            return "error: old_string must not be empty"
        n = before.count(old_string)
        if n != 1:
            return f"error: old_string occurs {n} times in {fp}; expected exactly 1"
        after = before.replace(old_string, content, 1)
    else:
        after = content
    if before == after and fp.is_file():
        return f"no changes: {fp}"
    if on_preview:
        on_preview(_write_stats(fp, before, after))
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(after, encoding="utf-8")
    return f"wrote {len(after.encode('utf-8'))} bytes to {fp}"

def tool_bash(cwd: Path, command: str, timeout: int | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout or 60, env=env,
        )
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

def tool_grep(cwd: Path, pattern: str, path: str | None = None, glob: str | None = None) -> str:
    target = str(_resolve(cwd, path) if path else cwd)
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never", pattern, target]
        if glob:
            cmd[1:1] = ["--glob", glob]
    else:
        cmd = ["grep", "-RIn", f"--include={glob}", "--", pattern, target] if glob else [
            "grep", "-RIn", "--", pattern, target
        ]
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    except subprocess.TimeoutExpired:
        return "error: grep timed out"
    except FileNotFoundError:
        return "error: neither rg nor grep available"
    out = proc.stdout or ""
    if proc.returncode not in (0, 1) and not out:
        return _truncate(proc.stderr or f"grep failed exit={proc.returncode}", MAX_GREP_CHARS)
    lines = out.splitlines()
    if len(lines) > MAX_GREP_LINES:
        out = "\n".join(lines[:MAX_GREP_LINES]) + f"\n...[{len(lines) - MAX_GREP_LINES} more]"
    return _truncate(out or "(no matches)", MAX_GREP_CHARS)

HANDLERS = {"read": tool_read, "write": tool_write, "bash": tool_bash, "grep": tool_grep}

def dispatch_tool(
    cwd: Path, name: str, args: dict[str, Any], *, on_preview: Callable[[str], None] | None = None
) -> str:
    fn = HANDLERS.get(name)
    if not fn:
        return f"error: unknown tool {name}"
    try:
        return tool_write(cwd, **args, on_preview=on_preview) if name == "write" else fn(cwd, **args)
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
        action = "replace" if args.get("old_string") is not None else "write"
        return f"{args.get('path', '')}  ({action}, {len(str(args.get('content') or ''))} chars)"
    if name == "grep":
        bits = [repr(args.get("pattern") or ""), "in", str(args.get("path") or ".")]
        if args.get("glob"):
            bits.append(str(args["glob"]))
        return _trunc(" ".join(bits))
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
) -> str:
    cwd = cwd.resolve()
    messages.append({"role": "user", "content": task})
    last_prompt = prompt_tokens
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
                            dispatch_tool, cwd, name, args, on_preview=on_preview
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

"""Async agent loop and tools: read, write, bash, grep."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

MAX_TOOL_CHARS = 50_000
MAX_TURNS = 30
MAX_GREP_LINES = 200

SYSTEM = """You are a coding agent. You can only use these tools: read, write, bash, grep.
Explore with read/grep before editing. Use bash to run tests and verify changes.
Prefer small, focused edits. When done, reply with a short summary — no more tool calls."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file (numbered lines). Optional 1-based offset/limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write/overwrite a text file (creates parent dirs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search with ripgrep (or grep -R fallback).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def _truncate(s: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated {len(s) - limit} chars]"


def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def tool_read(cwd: Path, path: str, offset: int | None = None, limit: int | None = None) -> str:
    fp = _resolve(cwd, path)
    if not fp.is_file():
        return f"error: not a file: {fp}"
    try:
        lines = fp.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"error: binary or non-utf8: {fp}"
    start = max(0, (offset or 1) - 1) if offset is not None else 0
    end = len(lines) if limit is None else start + max(0, limit)
    chunk = lines[start:end]
    return _truncate(
        "\n".join(f"{i + start + 1}|{line}" for i, line in enumerate(chunk)) or "(empty)"
    )


def tool_write(cwd: Path, path: str, content: str) -> str:
    fp = _resolve(cwd, path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {fp}"


def tool_bash(cwd: Path, command: str, timeout: int | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or 60,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return _truncate(f"timeout after {timeout or 60}s\n{(e.stdout or '')}{(e.stderr or '')}")
    parts = [f"exit={proc.returncode}"]
    if proc.stdout:
        parts.append("stdout:\n" + proc.stdout)
    if proc.stderr:
        parts.append("stderr:\n" + proc.stderr)
    return _truncate("\n".join(parts))


def tool_grep(
    cwd: Path, pattern: str, path: str | None = None, glob: str | None = None
) -> str:
    target = str(_resolve(cwd, path) if path else cwd)
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never", pattern, target]
        if glob:
            cmd[1:1] = ["--glob", glob]
    else:
        cmd = ["grep", "-RIn", "--", pattern, target]
        if glob:
            cmd = ["grep", "-RIn", f"--include={glob}", "--", pattern, target]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30
        )
    except subprocess.TimeoutExpired:
        return "error: grep timed out"
    except FileNotFoundError:
        return "error: neither rg nor grep available"
    out = proc.stdout or ""
    if proc.returncode not in (0, 1) and not out:
        return _truncate(proc.stderr or f"grep failed exit={proc.returncode}")
    lines = out.splitlines()
    if len(lines) > MAX_GREP_LINES:
        out = "\n".join(lines[:MAX_GREP_LINES]) + f"\n...[{len(lines) - MAX_GREP_LINES} more]"
    return _truncate(out or "(no matches)")


HANDLERS: dict[str, Callable[..., str]] = {
    "read": tool_read,
    "write": tool_write,
    "bash": tool_bash,
    "grep": tool_grep,
}


def dispatch_tool(cwd: Path, name: str, args: dict[str, Any]) -> str:
    fn = HANDLERS.get(name)
    if not fn:
        return f"error: unknown tool {name}"
    try:
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
        if off is not None or lim is not None:
            return f"{path}  (offset={off} limit={lim})"
        return path
    if name == "write":
        return f"{args.get('path', '')}  ({len(str(args.get('content') or ''))} bytes)"
    if name == "grep":
        bits = [repr(args.get("pattern") or ""), "in", str(args.get("path") or ".")]
        if args.get("glob"):
            bits.append(str(args["glob"]))
        return _trunc(" ".join(bits))
    return _trunc(json.dumps(args, ensure_ascii=False))


def _delta_reasoning(delta: Any) -> str | None:
    text = getattr(delta, "reasoning_content", None)
    if text:
        return text
    extra = getattr(delta, "model_extra", None) or {}
    if isinstance(extra, dict) and extra.get("reasoning_content"):
        return str(extra["reasoning_content"])
    return None


async def _emit(cb: Callable[..., Any] | None, *args: Any) -> None:
    if not cb:
        return
    maybe = cb(*args)
    if asyncio.iscoroutine(maybe):
        await maybe


OnText = Callable[[str], Any]
OnTool = Callable[[int, str, dict[str, Any], str], Any]


OnUsage = Callable[[dict[str, int]], Any]


def _usage_from_chunk(chunk: Any) -> dict[str, int] | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        return None
    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt_tokens"] = int(prompt)
    if completion is not None:
        out["completion_tokens"] = int(completion)
    if total is not None:
        out["total_tokens"] = int(total)
    return out or None


async def _stream_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    on_delta: OnText | None,
    on_thinking: OnText | None,
    on_answer_to_thinking: Callable[[], Any] | None,
) -> tuple[str, list[dict[str, Any]], dict[str, int] | None]:
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        stream = await client.chat.completions.create(**create_kwargs)
    except Exception:
        create_kwargs.pop("stream_options", None)
        stream = await client.chat.completions.create(**create_kwargs)
    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    saw_tools = False
    usage: dict[str, int] | None = None

    async for chunk in stream:
        maybe = _usage_from_chunk(chunk)
        if maybe:
            usage = maybe
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = _delta_reasoning(delta)
        if reasoning:
            await _emit(on_thinking, reasoning)
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
            content_parts.append(delta.content)
            await _emit(on_thinking if saw_tools else on_delta, delta.content)

    tool_calls = [
        {
            "id": tool_acc[i]["id"],
            "type": "function",
            "function": {
                "name": tool_acc[i]["name"],
                "arguments": tool_acc[i]["arguments"] or "{}",
            },
        }
        for i in sorted(tool_acc)
    ]
    return "".join(content_parts), tool_calls, usage


async def run_agent(
    task: str,
    *,
    cwd: Path,
    client: AsyncOpenAI,
    model: str,
    max_turns: int = MAX_TURNS,
    messages: list[dict[str, Any]],
    on_delta: OnText | None = None,
    on_thinking: OnText | None = None,
    on_answer_to_thinking: Callable[[], Any] | None = None,
    on_tool: OnTool | None = None,
    on_usage: OnUsage | None = None,
) -> str:
    cwd = cwd.resolve()
    messages.append({"role": "user", "content": task})

    for turn in range(1, max_turns + 1):
        content, tool_calls, usage = await _stream_completion(
            client,
            model=model,
            messages=messages,
            on_delta=on_delta,
            on_thinking=on_thinking,
            on_answer_to_thinking=on_answer_to_thinking,
        )
        if usage:
            await _emit(on_usage, usage)
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            return content

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
                result = f"error: invalid JSON arguments: {raw_args!r}"
            else:
                result = await asyncio.to_thread(dispatch_tool, cwd, name, args)
            if on_tool:
                await _emit(on_tool, turn, name, args, result)
            else:
                print(f"  [{turn}] {name} {tool_summary(name, args)}", file=sys.stderr, flush=True)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return f"(stopped after {max_turns} turns)"

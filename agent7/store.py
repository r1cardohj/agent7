"""One JSON snapshot per workspace."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import Session

VERSION = 2
READABLE_VERSIONS = frozenset({1, 2})  # 1: chat history; 2: same, plus Responses raw items

def session_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return root / "agent7" / "sessions"

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def workspace_key(cwd: Path) -> str:
    return hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]

def session_path(cwd: Path) -> Path:
    return session_dir() / f"{workspace_key(cwd)}.json"

def save(session: Session, *, created_at: str) -> Path:
    """Atomically replace the workspace snapshot. No id, no name, no history."""
    cwd = session.cwd.resolve()
    path = session_path(cwd)
    data = {
        "version": VERSION,
        "cwd": str(cwd),
        "model_ref": session.model_ref or session.model,
        "max_turns": session.max_turns,
        "created_at": created_at,
        "updated_at": timestamp(),
        "messages": session.messages,
    }
    root = path.parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        root.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=f".{path.stem}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path

def load(cwd: Path) -> dict[str, Any]:
    path = session_path(cwd)
    if not path.is_file():
        raise ValueError(f"no session for workspace: {cwd.resolve()}")
    data = _read_snapshot(path)
    if Path(data["cwd"]).expanduser().resolve() != cwd.resolve():
        raise ValueError(f"session belongs to another workspace: {path}")
    data["path"] = path
    return data

resume = load

def resume_messages(snapshot: dict[str, Any], system: str) -> list[dict[str, Any]]:
    """Refresh the prompt and repair an interrupted final group of tool calls."""
    saved = snapshot["messages"]
    rest = saved[1:] if saved and saved[0].get("role") == "system" else saved
    out = [{"role": "system", "content": system}, *rest]
    for i in range(len(out) - 1, 0, -1):
        calls = out[i].get("tool_calls") if out[i].get("role") == "assistant" else None
        if not isinstance(calls, list):
            continue
        answered = {m.get("tool_call_id") for m in out[i + 1:] if m.get("role") == "tool"}
        missing = []
        for call in calls:
            call_id = call.get("id") if isinstance(call, dict) else None
            if isinstance(call_id, str) and call_id and call_id not in answered:
                missing.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": "error: session interrupted before tool completion",
                })
        end = i + 1
        while end < len(out) and out[end].get("role") == "tool":
            end += 1
        out[end:end] = missing
        break
    if out[-1].get("role") == "tool" and not any(
        m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list) for m in out[1:]
    ):
        while out[-1].get("role") == "tool":
            out.pop()
    return out

def _read_snapshot(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"session not found: {path}") from None
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read session {path}: {e}") from None
    # Version 2 is what we write. Version 1 still loads; missing raw items are
    # replayed as portable calls. Anything else is refused so a newer snapshot
    # is not sent blindly.
    if not isinstance(data, dict) or data.get("version") not in READABLE_VERSIONS:
        raise ValueError(f"unsupported session format: {path}")
    if not isinstance(data.get("messages"), list):
        raise ValueError(f"invalid session snapshot: {path}")
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in data["messages"]):
        raise ValueError(f"invalid messages in session: {path}")
    if not isinstance(data.get("cwd"), str):
        raise ValueError(f"invalid workspace in session: {path}")
    for key in ("created_at", "updated_at"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"invalid {key} in session: {path}")
    if data.get("model_ref") is not None and not isinstance(data["model_ref"], str):
        raise ValueError(f"invalid model in session: {path}")
    if not isinstance(data.get("max_turns"), int) or data["max_turns"] <= 0:
        raise ValueError(f"invalid max_turns in session: {path}")
    return data

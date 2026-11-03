"""Small, workspace-scoped JSON session snapshots."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import Session

VERSION = 1
_ID = re.compile(r"[0-9a-f]{12}")

def session_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return root / "agent7" / "sessions"

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def session_path(session_id: str) -> Path:
    if not _ID.fullmatch(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    return session_dir() / f"{session_id}.json"

def save(
    session_id: str, session: Session, *, name: str = "", created_at: str,
) -> Path:
    """Atomically save only the state needed to resume a model conversation."""
    path = session_path(session_id)
    now = timestamp()
    data = {
        "version": VERSION,
        "id": session_id,
        "name": name,
        "cwd": str(session.cwd.resolve()),
        "model_ref": session.model_ref or session.model,
        "max_turns": session.max_turns,
        "created_at": created_at,
        "updated_at": now,
        "messages": session.messages,
    }
    root = path.parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        root.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=f".{session_id}.", suffix=".tmp")
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

def load(session_id: str, *, cwd: Path | None = None) -> dict[str, Any]:
    path = session_path(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"session not found: {session_id}") from None
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read session {session_id}: {e}") from None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise ValueError(f"unsupported session format: {path}")
    if data.get("id") != session_id or not isinstance(data.get("messages"), list):
        raise ValueError(f"invalid session snapshot: {path}")
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in data["messages"]):
        raise ValueError(f"invalid messages in session: {path}")
    if not isinstance(data.get("cwd"), str):
        raise ValueError(f"invalid workspace in session: {path}")
    for key in ("name", "created_at", "updated_at"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"invalid {key} in session: {path}")
    if data.get("model_ref") is not None and not isinstance(data["model_ref"], str):
        raise ValueError(f"invalid model in session: {path}")
    if not isinstance(data.get("max_turns"), int) or data["max_turns"] <= 0:
        raise ValueError(f"invalid max_turns in session: {path}")
    if cwd is not None and Path(data["cwd"]).expanduser().resolve() != cwd.resolve():
        raise ValueError(f"session {session_id} belongs to another workspace")
    data["path"] = path
    return data

def list_sessions(cwd: Path) -> list[dict[str, Any]]:
    root = session_dir()
    try:
        paths = list(root.glob("*.json"))
    except OSError:
        return []
    out = []
    for path in paths:
        try:
            out.append(load(path.stem, cwd=cwd))
        except ValueError:
            pass  # A broken or foreign snapshot must not hide the others.
    return sorted(out, key=lambda s: (s["updated_at"], s["id"]), reverse=True)

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

def resume(cwd: Path, selector: str = "latest") -> dict[str, Any]:
    if selector != "latest":
        return load(selector, cwd=cwd)
    sessions = list_sessions(cwd)
    if not sessions:
        raise ValueError(f"no sessions for workspace: {cwd.resolve()}")
    return sessions[0]

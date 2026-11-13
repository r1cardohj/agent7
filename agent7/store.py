"""One JSON snapshot per session, with workspace-scoped discovery."""

from __future__ import annotations

import hashlib
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

VERSION = 2
READABLE_VERSIONS = frozenset({1, 2})  # 1: chat history; 2: same, plus Responses raw items
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def session_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return root / "agent7" / "sessions"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def workspace_key(cwd: Path) -> str:
    return hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]


def new_session_id() -> str:
    """A globally unique, filename-safe ID that is unambiguous on the CLI."""
    return f"sess-{uuid.uuid4().hex}"


def legacy_session_id(cwd: Path) -> str:
    """A stable ID for a snapshot written before sessions had identities."""
    return f"legacy-{workspace_key(cwd)}"


def legacy_session_path(cwd: Path) -> Path:
    return session_dir() / f"{workspace_key(cwd)}.json"


def trash_session_dir() -> Path:
    return session_dir().parent / "trash"


def _id_path(session_id: str) -> Path:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    return session_dir() / f"{session_id}.json"


def session_path(cwd: Path, session_id: str | None = None) -> Path:
    """Return a session's path, or the latest/legacy path for old callers.

    New code should always pass an ID. With no ID, retaining the legacy answer
    when no modern snapshot exists also gives older snapshots a migration path.
    """
    if session_id is not None:
        return _id_path(session_id)
    snapshots = list_sessions(cwd)
    return Path(snapshots[0]["path"]) if snapshots else legacy_session_path(cwd)


def session_name(messages: list[dict[str, Any]], *, limit: int = 60) -> str:
    """Make a readable, single-line name from the first real user message."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        name = " ".join(content.split())
        if name:
            return name if len(name) <= limit else name[:limit - 1].rstrip() + "…"
    return ""


def save(session: Session, *, created_at: str | None = None) -> Path:
    """Atomically replace this session's snapshot, never its workspace's."""
    cwd = session.cwd.resolve()
    session_id = getattr(session, "id", "") or new_session_id()
    session.id = session_id
    created = created_at or getattr(session, "created_at", "") or timestamp()
    session.created_at = created
    derived_name = session_name(session.messages)
    name = derived_name or getattr(session, "name", "")
    session.name = name
    path = session_path(cwd, session_id)
    data = {
        "version": VERSION,
        "id": session_id,
        "name": name or "New session",
        "cwd": str(cwd),
        "model_ref": session.model_ref or session.model,
        "max_turns": session.max_turns,
        "created_at": created,
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
    session.persisted = True
    # A resumed legacy snapshot gets its real per-session file on the first
    # save. Remove the old key only after the replacement is durable, so both
    # the picker and `-r <legacy-id>` continue to refer to one session.
    legacy = legacy_session_path(cwd)
    if session_id == legacy_session_id(cwd) and path != legacy:
        with suppress(OSError):
            legacy.unlink()
    return path


def _prepare_snapshot(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Add identity fields without rewriting old files during discovery."""
    cwd = Path(data["cwd"]).expanduser().resolve()
    data.setdefault("id", legacy_session_id(cwd))
    data["name"] = data.get("name") or session_name(data["messages"]) or "Previous session"
    data["path"] = path
    return data


def _load_path(path: Path, cwd: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"no session for workspace: {cwd.resolve()}")
    data = _prepare_snapshot(path, _read_snapshot(path))
    if Path(data["cwd"]).expanduser().resolve() != cwd.resolve():
        raise ValueError(f"session belongs to another workspace: {path}")
    return data


def _scan_sessions(cwd: Path, *, strict: bool) -> list[dict[str, Any]]:
    root = session_dir()
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    found: list[dict[str, Any]] = []
    errors: list[ValueError] = []
    for path in paths:
        try:
            data = _load_path(path, cwd)
        except ValueError as e:
            # A picker should survive an unrelated corrupt snapshot. A direct
            # latest-resume still reports corruption when nothing else loads.
            if "belongs to another workspace" not in str(e):
                errors.append(e)
            continue
        found.append(data)
    found.sort(key=lambda data: (data["updated_at"], str(data["path"])), reverse=True)
    if strict and not found and errors:
        raise errors[0]
    return found


def list_sessions(cwd: Path) -> list[dict[str, Any]]:
    """All readable sessions in this workspace, most recently updated first."""
    return _scan_sessions(cwd, strict=False)


def delete_session(cwd: Path, session_id: str) -> Path:
    """Move one workspace session to trash without overwriting an old copy."""
    snapshot = load(cwd, session_id)
    source = Path(snapshot["path"])
    root = trash_session_dir()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        root.chmod(0o700)
    target = root / f"{session_id}.json"
    if target.exists():
        raise ValueError(f"a trashed session already has this id: {session_id}")
    try:
        source.replace(target)
    except OSError as e:
        raise ValueError(f"cannot delete session {source}: {e}") from None
    return target


def load(cwd: Path, session_id: str | None = None) -> dict[str, Any]:
    if session_id is None:
        snapshots = _scan_sessions(cwd, strict=True)
        if not snapshots:
            raise ValueError(f"no session for workspace: {cwd.resolve()}")
        return snapshots[0]
    path = session_path(cwd, session_id)
    if not path.is_file() and session_id == legacy_session_id(cwd):
        path = legacy_session_path(cwd)
    data = _load_path(path, cwd)
    if data["id"] != session_id:
        raise ValueError(f"session id mismatch: {path}")
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
    # replayed as portable calls. Identity fields are additive, not a format 3.
    if not isinstance(data, dict) or data.get("version") not in READABLE_VERSIONS:
        raise ValueError(f"unsupported session format: {path}")
    if not isinstance(data.get("messages"), list):
        raise ValueError(f"invalid session snapshot: {path}")
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in data["messages"]):
        raise ValueError(f"invalid messages in session: {path}")
    if not isinstance(data.get("cwd"), str):
        raise ValueError(f"invalid workspace in session: {path}")
    if "id" in data and (
        not isinstance(data["id"], str) or not SESSION_ID_PATTERN.fullmatch(data["id"])
    ):
        raise ValueError(f"invalid session id in session: {path}")
    if "name" in data and not isinstance(data["name"], str):
        raise ValueError(f"invalid name in session: {path}")
    for key in ("created_at", "updated_at"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"invalid {key} in session: {path}")
    if data.get("model_ref") is not None and not isinstance(data["model_ref"], str):
        raise ValueError(f"invalid model in session: {path}")
    if not isinstance(data.get("max_turns"), int) or data["max_turns"] <= 0:
        raise ValueError(f"invalid max_turns in session: {path}")
    return data

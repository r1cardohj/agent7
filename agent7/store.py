"""agent7 path policy for its workspace-scoped JSON session store."""

import os
from pathlib import Path
from typing import Any

from . import file_sessions as _file_sessions
from .file_sessions import FileSessionStore

VERSION = _file_sessions.VERSION
READABLE_VERSIONS = _file_sessions.READABLE_VERSIONS
new_session_id = _file_sessions.new_session_id
legacy_session_id = _file_sessions.legacy_session_id
session_name = _file_sessions.session_name
timestamp = _file_sessions.timestamp
workspace_key = _file_sessions.workspace_key


def session_dir() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return root / "agent7" / "sessions"


def trash_session_dir() -> Path:
    return session_dir().parent / "trash"


def _store() -> FileSessionStore:
    return FileSessionStore(session_dir(), trash_root=trash_session_dir())


def legacy_session_path(cwd: Path) -> Path:
    return _file_sessions.legacy_session_path(cwd, root=session_dir())


def session_path(cwd: Path, session_id: str | None = None) -> Path:
    return _store().session_path(cwd, session_id)


def save(session: Any, *, created_at: str | None = None) -> Path:
    return _store().save(session, created_at=created_at)


def list_sessions(cwd: Path) -> list[dict[str, Any]]:
    return _store().list(cwd)


def delete_session(cwd: Path, session_id: str) -> Path:
    return _store().delete(cwd, session_id)


def load(cwd: Path, session_id: str | None = None) -> dict[str, Any]:
    return _store().load(cwd, session_id)


resume = load


def resume_messages(snapshot: dict[str, Any], system: str) -> list[dict[str, Any]]:
    return _file_sessions.resume_messages(snapshot, system)

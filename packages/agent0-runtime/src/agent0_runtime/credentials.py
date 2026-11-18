"""Small JSON credential store with atomic writes and owner-only permissions."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .providers import Credential


class FileCredentialStore:
    """Store provider credentials in a host-selected JSON file.

    The path may be dynamic so embedding applications can honor their own
    environment and XDG configuration policy without runtime dependencies.
    """

    def __init__(self, path: Path | Callable[[], Path], *, source: str = "stored") -> None:
        self._path = path
        self.source = source

    @property
    def path(self) -> Path:
        return self._path() if callable(self._path) else self._path

    def load(self) -> dict[str, Any]:
        path = self.path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, data: dict[str, Any]) -> Path:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        tmp.replace(path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        return path

    def resolve(self, provider_id: str, default_base_url: str) -> Credential | None:
        entry = self.load().get(provider_id)
        if not isinstance(entry, dict) or entry.get("type") == "oauth":
            return None
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            return None
        base_url = entry.get("baseUrl")
        return Credential(
            key.strip(),
            base_url if isinstance(base_url, str) else default_base_url,
            self.source,
        )

    def login(self, provider_id: str, key: str, base_url: str) -> Credential:
        data = self.load()
        data[provider_id] = {"type": "api_key", "key": key, "baseUrl": base_url}
        data["active"] = provider_id
        self.save(data)
        return Credential(key, base_url, "login")

    def logout(self, provider_id: str) -> bool:
        data = self.load()
        if provider_id not in data:
            return False
        del data[provider_id]
        if data.get("active") == provider_id:
            data.pop("active", None)
        self.save(data)
        return True

    def active_provider(self) -> str:
        active = self.load().get("active")
        return active if isinstance(active, str) else ""

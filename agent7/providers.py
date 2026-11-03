"""Providers: paste API key → OpenAI-compatible client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

def _auth_path() -> Path:
    home = os.environ.get("AGENT7_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "auth.json"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() / "agent7" if xdg else Path.home() / ".config" / "agent7"
    return root / "auth.json"

agent7_auth_path = _auth_path

def load_auth() -> dict[str, Any]:
    path = _auth_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def _save(data: dict[str, Any]) -> Path:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path

@dataclass
class Credential:
    api_key: str
    base_url: str
    source: str

@dataclass
class ApiKeyProvider:
    id: str
    aliases: tuple[str, ...]
    env_keys: tuple[str, ...]
    default_base: str
    prompt: str
    default_model: str = ""
    login_name: str = ""

    @property
    def preferred_login(self) -> str:
        return self.login_name or self.id

    def resolve(self) -> Credential | None:
        base = os.environ.get("AGENT7_BASE_URL", "").strip() or self.default_base
        for env in self.env_keys:
            if key := os.environ.get(env, "").strip():
                return Credential(key, base, "env")
        entry = load_auth().get(self.id)
        if isinstance(entry, dict) and entry.get("type") != "oauth":
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                bu = entry.get("baseUrl") if isinstance(entry.get("baseUrl"), str) else self.default_base
                return Credential(key.strip(), bu, "agent7")
        return None

    async def login(self, *, args: list[str], prompt_secret: Any, on_status: Any | None = None) -> Credential:
        key = (prompt_secret(self.prompt) or "").strip()
        if not key:
            raise ValueError("cancelled — empty key")
        base = self.default_base
        data = load_auth()
        data[self.id] = {"type": "api_key", "key": key, "baseUrl": base}
        data["active"] = self.id
        _save(data)
        return Credential(key, base, "login")

    def logout(self) -> bool:
        data = load_auth()
        if self.id not in data:
            return False
        del data[self.id]
        if data.get("active") == self.id:
            data.pop("active", None)
        _save(data)
        return True

    async def list_models(self, cred: Credential) -> list[str]:
        client = self.make_client(cred)
        try:
            page = await client.models.list()
            return sorted({m.id for m in page.data if getattr(m, "id", None)})
        finally:
            await client.close()

    def make_client(self, cred: Credential) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=cred.api_key, base_url=cred.base_url)

_BUILTINS = [
    ApiKeyProvider("deepseek", ("ds",), ("DEEPSEEK_API_KEY",),
                   "https://api.deepseek.com", "DeepSeek API key", "deepseek-chat"),
    ApiKeyProvider("xai", ("grok",), ("XAI_API_KEY", "GROK_API_KEY"),
                   "https://api.x.ai/v1", "xAI API key (grok)", "grok-4", "grok"),
]
_REGISTRY: dict[str, ApiKeyProvider] = {}
for _p in _BUILTINS:
    _REGISTRY[_p.id] = _p
    for _a in _p.aliases:
        _REGISTRY[_a] = _p

def get_provider(name: str) -> ApiKeyProvider | None:
    return _REGISTRY.get(name.lower().replace("_", "-"))

def default_model_for(provider_id: str | None) -> str:
    """Fallback model when a provider's /models list is unavailable."""
    p = get_provider(provider_id or "")
    return p.default_model if p and p.default_model else _BUILTINS[0].default_model

def list_providers() -> list[ApiKeyProvider]:
    return list(_BUILTINS)

def resolve_all() -> dict[str, Credential]:
    return {p.id: c for p in _BUILTINS if (c := p.resolve())}

def resolve_default() -> tuple[ApiKeyProvider, Credential] | None:
    active = load_auth().get("active")
    if isinstance(active, str):
        p = get_provider(active)
        if p and (c := p.resolve()):
            return p, c
    for p in _BUILTINS:
        if c := p.resolve():
            return p, c
    return None

def login_hints() -> str:
    return " | ".join(f"/login {p.preferred_login}" for p in _BUILTINS)

def describe_found() -> list[tuple[str, str]]:
    active = load_auth().get("active")
    return [
        (f"{'*' if active == p.id else ' '} {p.id}", c.source)
        for p in _BUILTINS if (c := p.resolve())
    ]

Provider = ApiKeyProvider

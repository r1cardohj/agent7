"""agent7 auth/UI adapter over the generic provider metadata in core."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

from . import __version__
from .core.providers import BUILTIN_PROVIDERS, Credential, ProviderSpec


class Agent7Client(AsyncOpenAI):
    """OpenAI-compatible client identified as agent7, not the generic SDK."""

    @property
    def user_agent(self) -> str:
        return f"agent7/{__version__}"


class Agent7SyncClient(OpenAI):
    @property
    def user_agent(self) -> str:
        return f"agent7/{__version__}"


def make_client(api_key: str, base_url: str) -> Agent7Client:
    return Agent7Client(api_key=api_key, base_url=base_url)


def make_sync_client(api_key: str, base_url: str, *, timeout: float) -> Agent7SyncClient:
    return Agent7SyncClient(api_key=api_key, base_url=base_url, timeout=timeout)


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
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    tmp.replace(path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


class ApiKeyProvider(ProviderSpec):
    """agent7 login/auth-file behavior for a core provider spec."""

    def resolve(self) -> Credential | None:
        if credential := self.resolve_env():
            base_url = os.environ.get("AGENT7_BASE_URL", "").strip() or credential.base_url
            return Credential(credential.api_key, base_url, credential.source)
        entry = load_auth().get(self.id)
        if isinstance(entry, dict) and entry.get("type") != "oauth":
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                base_url = (
                    entry.get("baseUrl") if isinstance(entry.get("baseUrl"), str) else self.base_url
                )
                return Credential(key.strip(), base_url, "agent7")
        return None

    async def login(
        self, *, _args: list[str], prompt_secret: Any, _on_status: Any | None = None
    ) -> Credential:
        key = (prompt_secret(self.prompt) or "").strip()
        if not key:
            raise ValueError("cancelled — empty key")
        data = load_auth()
        data[self.id] = {"type": "api_key", "key": key, "baseUrl": self.base_url}
        data["active"] = self.id
        _save(data)
        return Credential(key, self.base_url, "login")

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

    def list_models_sync(self, cred: Credential, *, timeout: float) -> list[str]:
        client = make_sync_client(cred.api_key, cred.base_url, timeout=timeout)
        try:
            page = client.models.list()
            return sorted({m.id for m in page.data if getattr(m, "id", None)})
        finally:
            client.close()

    def make_client(self, cred: Credential) -> Agent7Client:
        return make_client(cred.api_key, cred.base_url)


_BUILTINS = [
    ApiKeyProvider(
        spec.id,
        spec.aliases,
        spec.env_keys,
        spec.base_url,
        spec.prompt,
        spec.default_model,
    )
    for spec in BUILTIN_PROVIDERS
]
_REGISTRY: dict[str, ApiKeyProvider] = {}
for _provider in _BUILTINS:
    _REGISTRY[_provider.id] = _provider
    for _alias in _provider.aliases:
        _REGISTRY[_alias] = _provider


def get_provider(name: str) -> ApiKeyProvider | None:
    return _REGISTRY.get(name.lower().replace("_", "-"))


def default_model_for(provider_id: str | None) -> str:
    provider = get_provider(provider_id or "")
    if provider and provider.default_model:
        return provider.default_model
    return _BUILTINS[0].default_model


def list_providers() -> list[ApiKeyProvider]:
    return list(_BUILTINS)


def resolve_all() -> dict[str, Credential]:
    return {p.id: c for p in _BUILTINS if (c := p.resolve())}


def resolve_default() -> tuple[ApiKeyProvider, Credential] | None:
    active = load_auth().get("active")
    if isinstance(active, str):
        provider = get_provider(active)
        if provider and (credential := provider.resolve()):
            return provider, credential
    for provider in _BUILTINS:
        if credential := provider.resolve():
            return provider, credential
    return None


def login_hints() -> str:
    return " | ".join(f"/login {p.preferred_login}" for p in _BUILTINS)


def describe_found() -> list[tuple[str, str]]:
    active = load_auth().get("active")
    return [
        (f"{'*' if active == p.id else ' '} {p.id}", c.source)
        for p in _BUILTINS
        if (c := p.resolve())
    ]


Provider = ApiKeyProvider

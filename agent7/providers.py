"""agent7 product configuration for runtime-managed providers and credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent0_runtime.client import (
    OpenAIClient,
    OpenAISyncClient,
    make_openai_client,
    make_openai_sync_client,
)
from agent0_runtime.credentials import FileCredentialStore
from agent0_runtime.managed import ManagedProvider
from agent0_runtime.providers import BUILTIN_PROVIDERS, Credential

from . import __version__

Agent7Client = OpenAIClient
Agent7SyncClient = OpenAISyncClient


def make_client(api_key: str, base_url: str) -> Agent7Client:
    return make_openai_client(
        api_key,
        base_url,
        application="agent7",
        version=__version__,
    )


def make_sync_client(api_key: str, base_url: str, *, timeout: float) -> Agent7SyncClient:
    return make_openai_sync_client(
        api_key,
        base_url,
        timeout=timeout,
        application="agent7",
        version=__version__,
    )


def _auth_path() -> Path:
    home = os.environ.get("AGENT7_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "auth.json"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() / "agent7" if xdg else Path.home() / ".config" / "agent7"
    return root / "auth.json"


agent7_auth_path = _auth_path
_STORE = FileCredentialStore(_auth_path, source="agent7")


def load_auth() -> dict[str, Any]:
    return _STORE.load()


def _save(data: dict[str, Any]) -> Path:
    return _STORE.save(data)


def _base_url_override() -> str:
    return os.environ.get("AGENT7_BASE_URL", "")


_BUILTINS = [
    ManagedProvider(
        spec.id,
        spec.aliases,
        spec.env_keys,
        spec.base_url,
        spec.prompt,
        spec.default_model,
        credential_store=_STORE,
        base_url_override=_base_url_override,
        client_factory=make_client,
        sync_client_factory=make_sync_client,
    )
    for spec in BUILTIN_PROVIDERS
]
_REGISTRY: dict[str, ManagedProvider] = {}
for _provider in _BUILTINS:
    _REGISTRY[_provider.id] = _provider
    for _alias in _provider.aliases:
        _REGISTRY[_alias] = _provider


def get_provider(name: str) -> ManagedProvider | None:
    return _REGISTRY.get(name.lower().replace("_", "-"))


def default_model_for(provider_id: str | None) -> str:
    provider = get_provider(provider_id or "")
    if provider and provider.default_model:
        return provider.default_model
    return _BUILTINS[0].default_model


def list_providers() -> list[ManagedProvider]:
    return list(_BUILTINS)


def resolve_all() -> dict[str, Credential]:
    return {p.id: c for p in _BUILTINS if (c := p.resolve())}


def resolve_default() -> tuple[ManagedProvider, Credential] | None:
    active = _STORE.active_provider()
    if active:
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
    active = _STORE.active_provider()
    return [
        (f"{'*' if active == p.id else ' '} {p.id}", c.source)
        for p in _BUILTINS
        if (c := p.resolve())
    ]


ApiKeyProvider = ManagedProvider
Provider = ManagedProvider

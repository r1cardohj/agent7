"""Provider-neutral model provider metadata and API-key resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Credential:
    api_key: str
    base_url: str
    source: str


@dataclass(frozen=True)
class ProviderSpec:
    """Static metadata for an OpenAI-compatible API provider."""

    id: str
    aliases: tuple[str, ...]
    env_keys: tuple[str, ...]
    base_url: str
    prompt: str
    default_model: str = ""
    login_name: str = ""

    @property
    def preferred_login(self) -> str:
        return self.login_name or self.id

    @property
    def default_base(self) -> str:
        """Compatibility spelling used by older agent7 callers."""

        return self.base_url

    def resolve_env(self) -> Credential | None:
        for env in self.env_keys:
            if key := os.environ.get(env, "").strip():
                return Credential(key, self.base_url, "env")
        return None


BUILTIN_PROVIDERS = (
    ProviderSpec(
        "deepseek",
        ("ds",),
        ("DEEPSEEK_API_KEY",),
        "https://api.deepseek.com",
        "DeepSeek API key",
        "deepseek-chat",
    ),
    ProviderSpec(
        "kimi",
        (),
        ("KIMI_API_KEY",),
        "https://api.kimi.com/coding/v1",
        "Kimi Code API key",
        "kimi-for-coding",
    ),
)

_REGISTRY: dict[str, ProviderSpec] = {}
for _provider in BUILTIN_PROVIDERS:
    _REGISTRY[_provider.id] = _provider
    for _alias in _provider.aliases:
        _REGISTRY[_alias] = _provider


def get_provider(name: str) -> ProviderSpec | None:
    return _REGISTRY.get(name.lower().replace("_", "-"))


def list_providers() -> list[ProviderSpec]:
    return list(BUILTIN_PROVIDERS)


def default_model_for(provider_id: str | None) -> str:
    """Fallback model when a provider's /models list is unavailable."""

    provider = get_provider(provider_id or "")
    if provider and provider.default_model:
        return provider.default_model
    return BUILTIN_PROVIDERS[0].default_model


def resolve_all_env() -> dict[str, Credential]:
    return {p.id: c for p in BUILTIN_PROVIDERS if (c := p.resolve_env())}

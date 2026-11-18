"""Provider behavior built from runtime metadata, credential storage, and clients."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .client import make_openai_client, make_openai_sync_client
from .credentials import FileCredentialStore
from .providers import Credential, ProviderSpec


@dataclass(frozen=True)
class ManagedProvider(ProviderSpec):
    """A ProviderSpec with optional host-selected storage and client factories."""

    credential_store: FileCredentialStore | None = None
    base_url_override: str | Callable[[], str] = ""
    client_factory: Callable[..., Any] = make_openai_client
    sync_client_factory: Callable[..., Any] = make_openai_sync_client

    def _override(self) -> str:
        value = (
            self.base_url_override() if callable(self.base_url_override) else self.base_url_override
        )
        return value.strip()

    def resolve(self) -> Credential | None:
        if credential := self.resolve_env():
            return Credential(
                credential.api_key,
                self._override() or credential.base_url,
                credential.source,
            )
        if self.credential_store is not None:
            return self.credential_store.resolve(self.id, self.base_url)
        return None

    async def login(
        self,
        *,
        _args: list[str],
        prompt_secret: Any,
        _on_status: Any | None = None,
    ) -> Credential:
        if self.credential_store is None:
            raise ValueError("no credential store is configured")
        key = (prompt_secret(self.prompt) or "").strip()
        if not key:
            raise ValueError("cancelled — empty key")
        return self.credential_store.login(self.id, key, self.base_url)

    def logout(self) -> bool:
        return self.credential_store.logout(self.id) if self.credential_store else False

    def make_client(self, credential: Credential) -> Any:
        return self.client_factory(credential.api_key, credential.base_url)

    async def list_models(self, credential: Credential) -> list[str]:
        client = self.make_client(credential)
        try:
            page = await client.models.list()
            return sorted({m.id for m in page.data if getattr(m, "id", None)})
        finally:
            await client.close()

    def list_models_sync(self, credential: Credential, *, timeout: float) -> list[str]:
        client = self.sync_client_factory(
            credential.api_key,
            credential.base_url,
            timeout=timeout,
        )
        try:
            page = client.models.list()
            return sorted({m.id for m in page.data if getattr(m, "id", None)})
        finally:
            client.close()

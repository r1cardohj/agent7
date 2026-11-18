"""Configurable OpenAI client factories with host-specific User-Agent metadata."""

from typing import Any

from openai import AsyncOpenAI, OpenAI


class OpenAIClient(AsyncOpenAI):
    """Async OpenAI client carrying an embedding application's identity."""

    def __init__(
        self, *args: Any, application: str = "agent0-runtime", version: str = "", **kwargs: Any
    ):
        self.application = application
        self.application_version = version
        super().__init__(*args, **kwargs)

    @property
    def user_agent(self) -> str:
        suffix = f"/{self.application_version}" if self.application_version else ""
        return f"{self.application}{suffix}"


class OpenAISyncClient(OpenAI):
    """Synchronous counterpart used for model discovery and host tooling."""

    def __init__(
        self, *args: Any, application: str = "agent0-runtime", version: str = "", **kwargs: Any
    ):
        self.application = application
        self.application_version = version
        super().__init__(*args, **kwargs)

    @property
    def user_agent(self) -> str:
        suffix = f"/{self.application_version}" if self.application_version else ""
        return f"{self.application}{suffix}"


def make_openai_client(
    api_key: str,
    base_url: str,
    *,
    application: str = "agent0-runtime",
    version: str = "",
) -> OpenAIClient:
    return OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        application=application,
        version=version,
    )


def make_openai_sync_client(
    api_key: str,
    base_url: str,
    *,
    timeout: float,
    application: str = "agent0-runtime",
    version: str = "",
) -> OpenAISyncClient:
    return OpenAISyncClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        application=application,
        version=version,
    )

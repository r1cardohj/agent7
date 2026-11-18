"""Host credential wiring for the runtime's built-in tools."""

from __future__ import annotations

from agent0_runtime.builtin_tools import BuiltinToolsConfig
from agent0_runtime.images import PillowImageReader, supports_vision

from .web_search import Agent7SearchProvider


def _deepseek_credential():
    from .providers import get_provider

    provider = get_provider("deepseek")
    return provider.resolve() if provider else None


def agent7_builtin_tools() -> BuiltinToolsConfig:
    from .config import SHELL  # local: avoids the module-level shell/config cycle

    return BuiltinToolsConfig(
        shell=SHELL,
        image_reader=PillowImageReader(),
        supports_vision=supports_vision,
        search_provider=Agent7SearchProvider(_deepseek_credential),
    )

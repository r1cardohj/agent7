"""Host credential wiring for the runtime's built-in tools."""

from __future__ import annotations

from agent0_runtime.builtin_tools import BuiltinToolsConfig
from agent0_runtime.images import PillowImageReader, supports_vision
from agent0_runtime.web_search import DeepSeekSearchProvider


def _deepseek_credential():
    from .providers import get_provider

    provider = get_provider("deepseek")
    return provider.resolve() if provider else None


def agent7_builtin_tools() -> BuiltinToolsConfig:
    from .agent import SHELL  # local: avoids an import cycle with the legacy facade

    return BuiltinToolsConfig(
        shell=SHELL,
        image_reader=PillowImageReader(),
        supports_vision=supports_vision,
        search_provider=DeepSeekSearchProvider(_deepseek_credential),
    )

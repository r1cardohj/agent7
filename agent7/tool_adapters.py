"""agent7 adapters for the generic runtime built-in tools."""

from __future__ import annotations

from agent0_runtime.builtin_tools import BuiltinToolsConfig
from agent0_runtime.images import PillowImageReader, supports_vision
from agent0_runtime.search import SearchProvider

from . import web_search


class DeepSeekSearchProvider(SearchProvider):
    """agent7's default DeepSeek Search API integration."""

    def search(
        self,
        query: str,
        count: int | None = None,
        *,
        user_question: str | None = None,
    ) -> str:
        return web_search.search(query, count, ori_question=user_question)


def agent7_builtin_tools() -> BuiltinToolsConfig:
    from .agent import SHELL  # local: avoids an import cycle with the legacy facade

    return BuiltinToolsConfig(
        shell=SHELL,
        image_reader=PillowImageReader(),
        supports_vision=supports_vision,
        search_provider=DeepSeekSearchProvider(),
    )

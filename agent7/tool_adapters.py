"""agent7 adapters for the generic runtime built-in tools."""

from __future__ import annotations

from . import web_search
from .core.builtin_tools import BuiltinToolsConfig
from .core.filesystem import ImageReader
from .core.search import SearchProvider
from .core.types import Attachment, ToolResult
from .images import read_image, supports_vision


class Agent7ImageReader(ImageReader):
    """Pillow-backed image processing, injected only into agent7's registry."""

    def read(self, path, _mime_type) -> ToolResult:
        result = read_image(path)
        return ToolResult(
            result.text,
            attachments=[
                Attachment(data=block.data, mime_type=block.mime_type, kind="image")
                for block in result.images
            ],
        )


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
        image_reader=Agent7ImageReader(),
        supports_vision=supports_vision,
        search_provider=DeepSeekSearchProvider(),
    )

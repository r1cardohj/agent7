"""Generic built-in tools, registered through the same public Tool SPI."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import web_fetch, web_search
from .filesystem import ImageReader, VisionSupport, edit_file, read_file, write_file
from .search import SearchProvider
from .shell import Shell, resolve_shell, run_shell
from .tools import Tool, ToolContext, ToolRegistry
from .types import ToolResult

DEFAULT_SEARCH_COUNT = 8
MIN_SEARCH_COUNT, MAX_SEARCH_COUNT = 1, 50
MAX_SEARCH_QUERY_CHARS = 128


@dataclass(frozen=True)
class BuiltinToolsConfig:
    """Application-provided services for the generic built-in tools."""

    shell: Shell = field(default_factory=resolve_shell)
    image_reader: ImageReader | None = None
    supports_vision: VisionSupport | None = None
    search_provider: SearchProvider | None = None


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_EDIT_ITEM = {
    "type": "object",
    "properties": {
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
    },
    "required": ["old_string", "new_string"],
}

BUILTIN_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _fn(
        "read",
        "Read text file (numbered). Default 200 lines; page with offset/limit. "
        "Also reads images (png/jpg/jpeg/gif/webp/bmp), which are returned as image attachments.",
        {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["path"],
    ),
    _fn(
        "write",
        "Create or overwrite a whole file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _fn(
        "edit",
        "Precise edits. One call may include multiple edits[] against the original "
        "file; no overlaps.",
        {"path": {"type": "string"}, "edits": {"type": "array", "items": _EDIT_ITEM}},
        ["path", "edits"],
    ),
    _fn(
        "bash",
        "Run a shell command in cwd (host shell). Output capped ~12k chars.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        ["command"],
    ),
    _fn(
        "fetch",
        "Fetch an http/https web page and extract readable Markdown text with metadata.",
        {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": web_fetch.MAX_FETCH_CHARS,
                "description": (
                    f"Maximum returned characters; defaults to {web_fetch.DEFAULT_FETCH_CHARS}."
                ),
            },
        },
        ["url"],
    ),
    _fn(
        "search",
        "Search the web (DeepSeek Search API) for ranked pages with summaries. "
        'The query takes site:, -site:, "...", lang:, after: and before: operators. '
        f"Read {web_search.skill_path()} before the first search of a task for "
        "query syntax and result triage.",
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": web_search.MAX_QUERY_CHARS,
                "description": "Search terms, optionally with search operators.",
            },
            "count": {
                "type": "integer",
                "minimum": web_search.MIN_COUNT,
                "maximum": web_search.MAX_COUNT,
                "description": (
                    f"Results to return; defaults to {web_search.DEFAULT_COUNT}. "
                    "Does not change the price of the request."
                ),
            },
        },
        ["query"],
    ),
)


def create_builtin_registry(config: BuiltinToolsConfig | None = None) -> ToolRegistry:
    """Create an instance-scoped registry of the generic built-in tools."""

    config = config or BuiltinToolsConfig()
    registry = ToolRegistry()

    def read(context: ToolContext, args: dict[str, Any]) -> str | ToolResult:
        return read_file(
            context.cwd,
            str(args.get("path") or ""),
            args.get("offset"),
            args.get("limit"),
            model=context.model,
            image_reader=config.image_reader,
            supports_vision=config.supports_vision,
        )

    def write(context: ToolContext, args: dict[str, Any]) -> str:
        return write_file(
            context.cwd,
            str(args.get("path") or ""),
            str(args.get("content") or ""),
            on_preview=context.preview,
        )

    def edit(context: ToolContext, args: dict[str, Any]) -> str:
        edits = args.get("edits")
        return edit_file(
            context.cwd,
            str(args.get("path") or ""),
            edits if isinstance(edits, list) else [],
            on_preview=context.preview,
        )

    def bash(context: ToolContext, args: dict[str, Any]) -> str:
        timeout = args.get("timeout")
        return run_shell(
            context.cwd,
            str(args.get("command") or ""),
            int(timeout) if isinstance(timeout, int) else None,
            shell=config.shell,
        )

    def fetch(_context: ToolContext, args: dict[str, Any]) -> str:
        max_chars = args.get("max_chars")
        return web_fetch.tool_fetch(
            _context.cwd,
            str(args.get("url") or ""),
            int(max_chars) if isinstance(max_chars, int) else None,
        )

    async def search(_context: ToolContext, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: query must be a non-empty string"
        if config.search_provider is None:
            return "error: no search provider is configured"
        count = args.get("count")
        result = config.search_provider.search(
            query,
            int(count) if isinstance(count, int) else None,
            user_question=_context.user_question,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    handlers: dict[str, Callable[[ToolContext, dict[str, Any]], Any]] = {
        "read": read,
        "write": write,
        "edit": edit,
        "bash": bash,
        "fetch": fetch,
        "search": search,
    }
    for schema in BUILTIN_TOOL_SCHEMAS:
        registry.register(Tool(schema=schema, handler=handlers[schema["function"]["name"]]))
    return registry

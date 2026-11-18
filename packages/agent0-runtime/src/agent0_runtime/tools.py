"""Instance-scoped tool definitions, registry, and serial async executor."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeAlias, Union, get_args, get_origin, get_type_hints

from .events import EventEmitter
from .types import AgentInterrupted, CancellationToken, ToolCall, ToolResult

ToolReturn: TypeAlias = str | ToolResult
ToolHandler: TypeAlias = Callable[
    ["ToolContext", dict[str, Any]], ToolReturn | Awaitable[ToolReturn]
]


@dataclass(frozen=True)
class Tool:
    """A Chat-function-compatible schema plus its runtime handler."""

    schema: dict[str, Any]
    handler: ToolHandler

    def __post_init__(self) -> None:
        function = self.schema.get("function")
        if self.schema.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("tool schema must be a function tool")
        if not str(function.get("name") or "").strip():
            raise ValueError("tool schema needs a function.name")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]

    @classmethod
    def define(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
        *,
        required: list[str] | None = None,
    ) -> Tool:
        """Define a function tool without hand-writing the outer schema."""

        return cls(
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required or [],
                    },
                },
            },
            handler=handler,
        )


_JSON_SCALAR_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _parameter_schema(annotation: Any) -> dict[str, Any]:
    """Translate common Python annotations into a function-tool JSON schema."""

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    origin = get_origin(annotation)
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin in (Union, UnionType):
        return {"anyOf": [_parameter_schema(item) for item in get_args(annotation)]}
    if annotation is list or origin is list:
        schema: dict[str, Any] = {"type": "array"}
        args = get_args(annotation)
        if args:
            schema["items"] = _parameter_schema(args[0])
        return schema
    if annotation is dict or origin is dict:
        return {"type": "object"}
    if annotation in _JSON_SCALAR_TYPES:
        return {"type": _JSON_SCALAR_TYPES[annotation]}
    raise TypeError(f"unsupported tool parameter annotation: {annotation!r}")


_TOOL_ATTRIBUTE = "__agent0_tool__"


def _as_tool(value: Any) -> Tool:
    """Resolve a Tool or a decorated function into its Tool definition."""

    if isinstance(value, Tool):
        return value
    tool_definition = getattr(value, _TOOL_ATTRIBUTE, None)
    if isinstance(tool_definition, Tool):
        return tool_definition
    raise TypeError("tools must be Tool instances or functions decorated with @tool")


def tool(
    name_or_function: str | Callable[..., Any] | None = None,
    *,
    name: str = "",
    description: str = "",
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a typed function as a Tool and keep it callable.

    ``@tool``, ``@tool(name="...")``, and ``@tool("...")`` all work. The name,
    JSON schema, required arguments, and docstring are stored on the returned
    function so an Agent can later resolve a decorated tool by its string name.
    """

    if isinstance(name_or_function, str):

        def configure(function: Callable[..., Any]) -> Callable[..., Any]:
            return _decorate_tool(function, name=name or name_or_function, description=description)

        return configure

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate_tool(function, name=name, description=description)

    return decorate if name_or_function is None else decorate(name_or_function)


def _decorate_tool(func: Callable[..., Any], *, name: str, description: str) -> Callable[..., Any]:
    """Build a Tool from a function and attach it to the function itself."""

    hints = get_type_hints(func)
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError("@tool functions may only declare named parameters")
        properties[parameter_name] = _parameter_schema(
            hints.get(parameter_name, parameter.annotation)
        )
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)

    tool_definition = Tool.define(
        name or func.__name__,
        description or inspect.getdoc(func) or func.__name__.replace("_", " "),
        properties,
        _tool_handler(func),
        required=required,
    )
    setattr(func, _TOOL_ATTRIBUTE, tool_definition)
    return func


def _tool_handler(
    func: Callable[..., Any],
) -> Callable[[ToolContext, dict[str, Any]], Any]:
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def invoke(_context: ToolContext, args: dict[str, Any]) -> Any:
            return await func(**args)

    else:

        @wraps(func)
        def invoke(_context: ToolContext, args: dict[str, Any]) -> Any:
            return func(**args)

    return invoke


class ToolRegistry:
    """Tools registered on one runtime, never process-global mutable handlers."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool | Callable[..., Any], *, replace_existing: bool = False) -> None:
        tool = _as_tool(tool)
        if tool.name in self._tools and not replace_existing:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]


@dataclass(frozen=True)
class ToolContext:
    """Host-neutral execution context supplied to every tool/plugin."""

    cwd: Path | None  # required only by filesystem/shell tools
    call: ToolCall
    model: str | None = None
    user_question: str | None = None
    preview: Callable[[str], None] | None = None
    cancellation: CancellationToken | None = None
    services: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecution:
    call: ToolCall
    args: dict[str, Any]
    result: ToolResult


class ToolExecutor:
    """Validate arguments, execute sync/async handlers, and emit tool events."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        events: EventEmitter,
        *,
        turn: int,
    ) -> ToolExecution:
        if context.cancellation is not None:
            context.cancellation.throw_if_cancelled(run_id=events.run_id)
        args: dict[str, Any] = {}
        try:
            decoded = json.loads(call.arguments or "{}")
            if not isinstance(decoded, dict):
                raise TypeError("expected a JSON object")
            args = decoded
            error: str | None = None
        except (json.JSONDecodeError, TypeError) as exc:
            error = f"error: invalid JSON arguments: {call.arguments!r} ({exc})"

        await events.emit(
            "tool_started",
            turn=turn,
            call_id=call.id,
            name=call.name,
            args=args,
        )
        if error is not None:
            result = ToolResult(error)
            await events.emit(
                "tool_finished",
                turn=turn,
                call_id=call.id,
                name=call.name,
                args=args,
                result=result.text,
            )
            return ToolExecution(call, args, result)

        tool = self.registry.get(call.name)
        if tool is None:
            result = ToolResult(f"error: unknown tool {call.name}")
            await events.emit(
                "tool_finished",
                turn=turn,
                call_id=call.id,
                name=call.name,
                args=args,
                result=result.text,
            )
            return ToolExecution(call, args, result)

        previews: list[str] = []
        local = replace(context, preview=previews.append)
        try:
            if inspect.iscoroutinefunction(tool.handler):
                returned = await tool.handler(local, args)
            else:
                returned = await asyncio.to_thread(tool.handler, local, args)
                if inspect.isawaitable(returned):
                    returned = await returned
            if isinstance(returned, ToolResult):
                result = returned
            elif isinstance(returned, str):
                result = ToolResult(returned)
            else:
                raise TypeError("tool handler must return str or ToolResult")
        except (AgentInterrupted, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except TypeError as exc:
            result = ToolResult(f"error: bad args for {call.name}: {exc}")
        except Exception as exc:
            result = ToolResult(f"error: {type(exc).__name__}: {exc}")

        for preview in previews:
            await events.emit(
                "tool_preview",
                turn=turn,
                call_id=call.id,
                name=call.name,
                args=args,
                text=preview,
            )
        if context.cancellation is not None:
            context.cancellation.throw_if_cancelled(run_id=events.run_id)
        await events.emit(
            "tool_finished",
            turn=turn,
            call_id=call.id,
            name=call.name,
            args=args,
            result=result.text,
        )
        return ToolExecution(call, args, result)

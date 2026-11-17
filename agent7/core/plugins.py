"""Small plugin SDK with explicit, allowlisted entry-point discovery."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from .backend import BackendRegistry
from .events import EventBus
from .lifecycle import LifecycleHooks
from .tools import ToolRegistry

PLUGIN_API_VERSION = 1
PLUGIN_ENTRY_POINT = "agent0.plugins"


@dataclass
class PluginContext:
    """Stable registration surface exposed to a plugin during application setup."""

    tools: ToolRegistry
    backends: BackendRegistry
    events: EventBus
    lifecycle: LifecycleHooks
    services: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Plugin(Protocol):
    api_version: int

    def register(self, context: PluginContext) -> None: ...


@dataclass(frozen=True)
class PluginInfo:
    """Discovered metadata; creating this object does not import plugin code."""

    name: str
    value: str
    distribution: str
    entry_point: Any = field(repr=False, compare=False)


class PluginError(RuntimeError):
    pass


def discover_plugins(group: str = PLUGIN_ENTRY_POINT) -> list[PluginInfo]:
    """List installed plugins without importing them."""

    try:
        points = metadata.entry_points().select(group=group)
    except AttributeError:  # Python 3.10's older EntryPoints API
        points = metadata.entry_points().get(group, ())
    found = [
        PluginInfo(
            name=point.name,
            value=point.value,
            distribution=getattr(getattr(point, "dist", None), "name", "") or "",
            entry_point=point,
        )
        for point in points
    ]
    return sorted(found, key=lambda item: (item.name, item.distribution, item.value))


class PluginManager:
    """Load trusted plugin objects and optional explicitly named entry points."""

    def __init__(self, context: PluginContext, *, group: str = PLUGIN_ENTRY_POINT) -> None:
        self.context = context
        self.group = group
        self._loaded: dict[str, Plugin] = {}
        self._order: list[str] = []

    @property
    def loaded(self) -> tuple[str, ...]:
        return tuple(self._order)

    def discover(self) -> list[PluginInfo]:
        return discover_plugins(self.group)

    def load(self, plugin: Plugin, *, name: str = "") -> str:
        plugin_name = name or str(
            getattr(plugin, "name", "") or f"{type(plugin).__module__}.{type(plugin).__qualname__}"
        )
        if plugin_name in self._loaded:
            raise PluginError(f"plugin already loaded: {plugin_name}")
        version = getattr(plugin, "api_version", None)
        if version != PLUGIN_API_VERSION:
            raise PluginError(
                f"plugin {plugin_name} uses API {version!r}; expected {PLUGIN_API_VERSION}"
            )
        register = getattr(plugin, "register", None)
        if not callable(register):
            raise PluginError(f"plugin {plugin_name} has no register(context) method")
        try:
            register(self.context)
        except Exception as error:
            raise PluginError(f"plugin {plugin_name} failed to register: {error}") from error
        self._loaded[plugin_name] = plugin
        self._order.append(plugin_name)
        return plugin_name

    def load_discovered(self, enabled: Iterable[str]) -> tuple[str, ...]:
        """Import only the explicitly allowlisted discovered plugin names."""

        discovered: dict[str, PluginInfo] = {}
        for info in self.discover():
            if info.name in discovered:
                raise PluginError(f"multiple installed plugins named {info.name}")
            discovered[info.name] = info
        loaded = []
        for name in enabled:
            info = discovered.get(name)
            if info is None:
                raise PluginError(f"enabled plugin is not installed: {name}")
            try:
                plugin = info.entry_point.load()
            except Exception as error:
                raise PluginError(f"cannot import plugin {name}: {error}") from error
            if isinstance(plugin, type):
                plugin = plugin()
            loaded.append(self.load(plugin, name=name))
        return tuple(loaded)

    async def close(self) -> None:
        """Close plugins in reverse registration order when they provide close()."""

        for name in reversed(self._order):
            plugin = self._loaded[name]
            close = getattr(plugin, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result

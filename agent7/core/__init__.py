"""The future agent0-runtime boundary, currently packaged inside agent7.

Nothing in this package imports the terminal, provider configuration, session
storage, or agent7's concrete tools. It can therefore be exercised with a fake
backend and moved to a standalone distribution without moving the UI as well.
"""

from .backend import BackendRegistry, ModelBackend, ModelRequest
from .events import (
    EventBus,
    EventEmitter,
    EventListener,
    ListenerError,
    RuntimeEvent,
    RuntimeEventKind,
)
from .lifecycle import (
    LifecycleEvent,
    LifecycleHooks,
    LifecycleKind,
    LifecycleListenerError,
    emit_lifecycle,
    run_with_lifecycle,
)
from .loop import DEFAULT_MAX_TURNS, Agent, HistoryPolicy
from .plugins import (
    PLUGIN_API_VERSION,
    PLUGIN_ENTRY_POINT,
    Plugin,
    PluginContext,
    PluginError,
    PluginInfo,
    PluginManager,
    discover_plugins,
)
from .tools import Tool, ToolContext, ToolExecution, ToolExecutor, ToolRegistry
from .types import (
    AgentInterrupted,
    CancellationToken,
    Message,
    ModelTurn,
    Recovery,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
    new_run_id,
)

__all__ = [
    "Agent",
    "AgentInterrupted",
    "BackendRegistry",
    "CancellationToken",
    "DEFAULT_MAX_TURNS",
    "EventBus",
    "EventEmitter",
    "EventListener",
    "HistoryPolicy",
    "LifecycleEvent",
    "LifecycleHooks",
    "LifecycleKind",
    "LifecycleListenerError",
    "ListenerError",
    "Message",
    "ModelBackend",
    "ModelRequest",
    "ModelTurn",
    "PLUGIN_API_VERSION",
    "PLUGIN_ENTRY_POINT",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "Recovery",
    "RunResult",
    "RunStatus",
    "RuntimeEvent",
    "RuntimeEventKind",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolExecution",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "discover_plugins",
    "emit_lifecycle",
    "new_run_id",
    "run_with_lifecycle",
]

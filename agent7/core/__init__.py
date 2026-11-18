"""The future agent0-runtime boundary, currently packaged inside agent7.

Nothing in this package imports the terminal, provider configuration, session
storage, or agent7's concrete tools. It can therefore be exercised with a fake
backend and moved to a standalone distribution without moving the UI as well.
"""

from .backend import BackendRegistry, ModelBackend, ModelInfo, ModelRequest
from .builtin_tools import (
    BUILTIN_TOOL_SCHEMAS,
    BuiltinToolsConfig,
    create_builtin_registry,
)
from .events import (
    EventBus,
    EventEmitter,
    EventListener,
    ListenerError,
    RuntimeEvent,
    RuntimeEventKind,
)
from .filesystem import ImageReader, VisionSupport
from .history import ResponsesHistoryPolicy
from .lifecycle import (
    LifecycleContext,
    LifecycleEvent,
    LifecycleHooks,
    LifecycleKind,
    LifecycleListenerError,
    emit_lifecycle,
    run_with_lifecycle,
)
from .loop import DEFAULT_MAX_TURNS, Agent, HistoryPolicy
from .openai_backend import OpenAIResponsesBackend
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
from .search import SearchProvider
from .session import MemorySessionStore, RunContext, RuntimeSession, SessionStore
from .shell import Shell
from .tools import Tool, ToolContext, ToolExecution, ToolExecutor, ToolRegistry
from .types import (
    TOOL_ATTACHMENTS_KEY,
    AgentInterrupted,
    Attachment,
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
    "Attachment",
    "BackendRegistry",
    "BUILTIN_TOOL_SCHEMAS",
    "BuiltinToolsConfig",
    "CancellationToken",
    "DEFAULT_MAX_TURNS",
    "EventBus",
    "EventEmitter",
    "EventListener",
    "HistoryPolicy",
    "ImageReader",
    "OpenAIResponsesBackend",
    "LifecycleContext",
    "LifecycleEvent",
    "LifecycleHooks",
    "LifecycleKind",
    "LifecycleListenerError",
    "ListenerError",
    "Message",
    "ModelBackend",
    "ModelInfo",
    "ModelRequest",
    "MemorySessionStore",
    "ModelTurn",
    "PLUGIN_API_VERSION",
    "PLUGIN_ENTRY_POINT",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "Recovery",
    "RunContext",
    "RunResult",
    "RunStatus",
    "ResponsesHistoryPolicy",
    "RuntimeEvent",
    "RuntimeEventKind",
    "RuntimeSession",
    "TOOL_ATTACHMENTS_KEY",
    "SearchProvider",
    "SessionStore",
    "Shell",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolExecution",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "VisionSupport",
    "create_builtin_registry",
    "discover_plugins",
    "emit_lifecycle",
    "new_run_id",
    "run_with_lifecycle",
]

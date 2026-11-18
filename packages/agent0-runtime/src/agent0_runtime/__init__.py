"""The standalone, embeddable agent0 runtime.

The backend/tool protocols, structured events, sessions, lifecycle, and plugins
are independent of terminal interfaces and application-owned policy.
"""

__version__ = "0.1.0"

from .backend import BackendRegistry, ModelBackend, ModelInfo, ModelRequest
from .builtin_tools import (
    BUILTIN_TOOL_SCHEMAS,
    BuiltinToolsConfig,
    create_builtin_registry,
)
from .capabilities import MODEL_CAPABILITIES, ModelCapability
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
from .images import PillowImageReader, supports_vision
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
from .providers import (
    BUILTIN_PROVIDERS,
    Credential,
    ProviderSpec,
    default_model_for,
    get_provider,
    list_providers,
    resolve_all_env,
)
from .search import SearchProvider
from .session import MemorySessionStore, RunContext, RuntimeSession, SessionStore
from .shell import Shell
from .tools import Tool, ToolContext, ToolExecution, ToolExecutor, ToolRegistry, tool
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
    "BUILTIN_PROVIDERS",
    "BUILTIN_TOOL_SCHEMAS",
    "BuiltinToolsConfig",
    "CancellationToken",
    "Credential",
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
    "ModelCapability",
    "MODEL_CAPABILITIES",
    "ModelInfo",
    "ModelRequest",
    "MemorySessionStore",
    "ModelTurn",
    "PLUGIN_API_VERSION",
    "PLUGIN_ENTRY_POINT",
    "Plugin",
    "ProviderSpec",
    "PluginContext",
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "PillowImageReader",
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
    "default_model_for",
    "discover_plugins",
    "emit_lifecycle",
    "get_provider",
    "list_providers",
    "new_run_id",
    "resolve_all_env",
    "run_with_lifecycle",
    "supports_vision",
    "tool",
]

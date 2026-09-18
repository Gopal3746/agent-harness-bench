"""Instrumented coding-agent evaluation harness."""

from agent_harness.llm import (
    ChatCompletionResult,
    OpenRouterClient,
    OpenRouterError,
    StreamedToolCall,
    TokenUsage,
)
from agent_harness.models import (
    BenchmarkConfig,
    ExecutionStrategy,
    TaskSpec,
)
from agent_harness.tools import (
    CommandResult,
    CommandRunner,
    RepositoryTools,
    SearchMatch,
    ToolDispatcher,
    ToolExecutionError,
    ToolResponse,
    WriteResult,
    build_tool_schemas,
)
from agent_harness.workspace import (
    TaskWorkspace,
    WorkspaceManager,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "ChatCompletionResult",
    "CommandResult",
    "CommandRunner",
    "ExecutionStrategy",
    "OpenRouterClient",
    "OpenRouterError",
    "RepositoryTools",
    "SearchMatch",
    "StreamedToolCall",
    "TaskSpec",
    "TaskWorkspace",
    "TokenUsage",
    "ToolDispatcher",
    "ToolExecutionError",
    "ToolResponse",
    "WorkspaceManager",
    "WriteResult",
    "__version__",
    "build_tool_schemas",
]

"""Instrumented coding-agent evaluation harness."""

from agent_harness.agent import (
    AgentRunResult,
    AgentStatus,
    ChatClient,
    SingleAgent,
    ToolCallRecord,
)
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
from agent_harness.runner import (
    SingleTaskRunner,
    TaskRunResult,
)
from agent_harness.telemetry import (
    JsonlRunWriter,
    ModelCallMetrics,
    RunRecord,
    ToolCallMetrics,
    VerificationMetrics,
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
    "AgentRunResult",
    "AgentStatus",
    "BenchmarkConfig",
    "ChatClient",
    "ChatCompletionResult",
    "CommandResult",
    "CommandRunner",
    "ExecutionStrategy",
    "JsonlRunWriter",
    "ModelCallMetrics",
    "OpenRouterClient",
    "OpenRouterError",
    "RepositoryTools",
    "RunRecord",
    "SearchMatch",
    "SingleAgent",
    "SingleTaskRunner",
    "StreamedToolCall",
    "TaskRunResult",
    "TaskSpec",
    "TaskWorkspace",
    "TokenUsage",
    "ToolCallMetrics",
    "ToolCallRecord",
    "ToolDispatcher",
    "ToolExecutionError",
    "ToolResponse",
    "VerificationMetrics",
    "WorkspaceManager",
    "WriteResult",
    "__version__",
    "build_tool_schemas",
]

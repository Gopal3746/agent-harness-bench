"""Instrumented coding-agent evaluation harness."""

from agent_harness.models import (
    BenchmarkConfig,
    ExecutionStrategy,
    TaskSpec,
)
from agent_harness.tools import (
    RepositoryTools,
    SearchMatch,
    ToolExecutionError,
)
from agent_harness.workspace import (
    TaskWorkspace,
    WorkspaceManager,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "ExecutionStrategy",
    "RepositoryTools",
    "SearchMatch",
    "TaskSpec",
    "TaskWorkspace",
    "ToolExecutionError",
    "WorkspaceManager",
    "__version__",
]

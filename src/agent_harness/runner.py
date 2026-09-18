from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_harness.agent import (
    AgentRunResult,
    ChatClient,
    SingleAgent,
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
    ToolDispatcher,
)
from agent_harness.workspace import (
    TaskWorkspace,
    WorkspaceManager,
)


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """Agent output and independent verification for one task run."""

    run_id: str
    task_id: str
    strategy: ExecutionStrategy
    model: str
    workspace: TaskWorkspace
    agent: AgentRunResult
    verification: CommandResult
    wall_time_ms: float

    @property
    def passed(self) -> bool:
        """Whether the agent completed and the verifier succeeded."""

        return self.agent.completed and self.verification.succeeded


class SingleTaskRunner:
    """Runs one task with the single-agent strategy."""

    def __init__(
        self,
        client: ChatClient,
        workspaces: WorkspaceManager,
        config: BenchmarkConfig,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if ExecutionStrategy.SINGLE not in config.strategies:
            raise ValueError(
                "benchmark config must include the single strategy"
            )

        self.client = client
        self.workspaces = workspaces
        self.config = config
        self.clock_ns = clock_ns

    def run(
        self,
        task: TaskSpec,
        *,
        run_id: str | None = None,
    ) -> TaskRunResult:
        """Execute and verify a task inside a fresh workspace."""

        started_ns = self.clock_ns()

        with self.workspaces.provision(
            task,
            run_id=run_id,
        ) as workspace:
            repository = RepositoryTools(workspace.path)
            commands = CommandRunner(workspace.path)
            dispatcher = ToolDispatcher(
                repository=repository,
                commands=commands,
                task=task,
            )
            agent = SingleAgent(
                self.client,
                dispatcher,
                model=self.config.model,
                max_steps=self.config.max_steps,
                temperature=self.config.temperature,
                seed=self.config.seed,
                session_id=workspace.run_id,
            )

            agent_result = agent.run(task)
            verification = commands.run_tests(task)

            return TaskRunResult(
                run_id=workspace.run_id,
                task_id=task.task_id,
                strategy=ExecutionStrategy.SINGLE,
                model=self.config.model,
                workspace=workspace,
                agent=agent_result,
                verification=verification,
                wall_time_ms=(
                    self.clock_ns() - started_ns
                )
                / 1_000_000,
            )

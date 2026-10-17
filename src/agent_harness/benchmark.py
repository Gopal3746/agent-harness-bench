from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from agent_harness.agent import ChatClient
from agent_harness.models import (
    BenchmarkConfig,
    ExecutionStrategy,
    TaskSpec,
)
from agent_harness.runner import TaskRunner, TaskRunResult
from agent_harness.telemetry import JsonlRunWriter
from agent_harness.workspace import WorkspaceManager


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Results collected for one controlled benchmark experiment."""

    experiment_id: str
    task_id: str
    runs: tuple[TaskRunResult, ...]

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def passed_runs(self) -> int:
        return sum(run.passed for run in self.runs)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0

        return self.passed_runs / self.total_runs

    def runs_for(
        self,
        strategy: ExecutionStrategy,
    ) -> tuple[TaskRunResult, ...]:
        """Return runs for one orchestration strategy."""

        return tuple(
            run for run in self.runs if run.strategy is strategy
        )


class BenchmarkRunner:
    """Executes every configured strategy and repetition."""

    def __init__(
        self,
        client: ChatClient,
        workspaces: WorkspaceManager,
        config: BenchmarkConfig,
        *,
        writer: JsonlRunWriter | None = None,
    ) -> None:
        self.config = config
        self.writer = writer
        self.task_runner = TaskRunner(
            client,
            workspaces,
            config,
        )

    def run(
        self,
        task: TaskSpec,
        *,
        experiment_id: str | None = None,
    ) -> BenchmarkResult:
        """Run the configured experiment for one coding task."""

        resolved_experiment_id = experiment_id or uuid4().hex[:12]
        results: list[TaskRunResult] = []

        for strategy in self.config.strategies:
            for repetition in range(1, self.config.repetitions + 1):
                result = self.task_runner.run(
                    task,
                    strategy=strategy,
                    run_id=self._run_id(
                        resolved_experiment_id,
                        task.task_id,
                        strategy,
                        repetition,
                    ),
                )
                results.append(result)

                if self.writer is not None:
                    self.writer.append_result(result)

        return BenchmarkResult(
            experiment_id=resolved_experiment_id,
            task_id=task.task_id,
            runs=tuple(results),
        )

    @staticmethod
    def _run_id(
        experiment_id: str,
        task_id: str,
        strategy: ExecutionStrategy,
        repetition: int,
    ) -> str:
        return (
            f"{experiment_id}-{task_id}-"
            f"{strategy.value}-{repetition:03d}"
        )

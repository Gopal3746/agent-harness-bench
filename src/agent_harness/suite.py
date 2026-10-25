from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent_harness.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
)
from agent_harness.models import TaskSpec


def _new_suite_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Results from benchmarking multiple coding tasks."""

    suite_id: str
    task_results: tuple[BenchmarkResult, ...]

    @property
    def total_tasks(self) -> int:
        return len(self.task_results)

    @property
    def total_runs(self) -> int:
        return sum(
            result.total_runs
            for result in self.task_results
        )

    @property
    def passed_runs(self) -> int:
        return sum(
            result.passed_runs
            for result in self.task_results
        )

    @property
    def failed_runs(self) -> int:
        return self.total_runs - self.passed_runs

    @property
    def pass_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0

        return self.passed_runs / self.total_runs

    @property
    def all_passed(self) -> bool:
        return (
            self.total_runs > 0
            and self.passed_runs == self.total_runs
        )


class BenchmarkSuiteRunner:
    """Coordinates benchmark runs across multiple tasks."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        *,
        id_factory: Callable[[], str] = _new_suite_id,
    ) -> None:
        self.runner = runner
        self.id_factory = id_factory

    def run(
        self,
        tasks: Sequence[TaskSpec],
        *,
        suite_id: str | None = None,
    ) -> BenchmarkSuiteResult:
        if not tasks:
            raise ValueError(
                "benchmark suite requires at least one task"
            )

        task_ids = [
            task.task_id
            for task in tasks
        ]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "benchmark suite contains duplicate task IDs"
            )

        resolved_suite_id = (
            suite_id
            if suite_id is not None
            else self.id_factory()
        )

        if not resolved_suite_id.strip():
            raise ValueError(
                "suite_id must not be empty"
            )

        task_results = tuple(
            self.runner.run(
                task,
                experiment_id=(
                    f"{resolved_suite_id}-{task.task_id}"
                ),
            )
            for task in tasks
        )

        return BenchmarkSuiteResult(
            suite_id=resolved_suite_id,
            task_results=task_results,
        )

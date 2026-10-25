from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_harness import (
    BenchmarkSuiteRunner,
    TaskSpec,
)


@dataclass(frozen=True)
class FakeBenchmarkResult:
    task_id: str
    total_runs: int
    passed_runs: int


class FakeBenchmarkRunner:
    def __init__(
        self,
        results: list[FakeBenchmarkResult],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    def run(
        self,
        task: TaskSpec,
        *,
        experiment_id: str,
    ) -> FakeBenchmarkResult:
        self.calls.append(
            (task.task_id, experiment_id)
        )
        return self.results.pop(0)


def make_task(
    tmp_path: Path,
    task_id: str,
) -> TaskSpec:
    source_dir = tmp_path / task_id
    source_dir.mkdir()

    return TaskSpec(
        task_id=task_id,
        instruction="Complete the task.",
        source_dir=source_dir,
        test_command=(
            "python",
            "-m",
            "pytest",
            "-q",
        ),
        timeout_seconds=30,
        tags=("suite",),
    )


def test_suite_runner_coordinates_all_tasks(
    tmp_path: Path,
) -> None:
    tasks = (
        make_task(tmp_path, "task-a"),
        make_task(tmp_path, "task-b"),
    )
    benchmark_runner = FakeBenchmarkRunner(
        [
            FakeBenchmarkResult(
                task_id="task-a",
                total_runs=6,
                passed_runs=5,
            ),
            FakeBenchmarkResult(
                task_id="task-b",
                total_runs=6,
                passed_runs=4,
            ),
        ]
    )

    result = BenchmarkSuiteRunner(
        benchmark_runner,
    ).run(
        tasks,
        suite_id="suite-001",
    )

    assert result.suite_id == "suite-001"
    assert result.total_tasks == 2
    assert result.total_runs == 12
    assert result.passed_runs == 9
    assert result.failed_runs == 3
    assert result.pass_rate == 0.75
    assert result.all_passed is False

    assert benchmark_runner.calls == [
        (
            "task-a",
            "suite-001-task-a",
        ),
        (
            "task-b",
            "suite-001-task-b",
        ),
    ]


def test_suite_runner_generates_suite_id(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path, "task-a")
    benchmark_runner = FakeBenchmarkRunner(
        [
            FakeBenchmarkResult(
                task_id="task-a",
                total_runs=1,
                passed_runs=1,
            )
        ]
    )

    result = BenchmarkSuiteRunner(
        benchmark_runner,
        id_factory=lambda: "generated-suite",
    ).run((task,))

    assert result.suite_id == "generated-suite"
    assert result.all_passed is True


def test_suite_runner_rejects_empty_or_duplicate_tasks(
    tmp_path: Path,
) -> None:
    runner = BenchmarkSuiteRunner(
        FakeBenchmarkRunner([])
    )

    with pytest.raises(
        ValueError,
        match="at least one task",
    ):
        runner.run(())

    task = make_task(tmp_path, "duplicate")

    with pytest.raises(
        ValueError,
        match="duplicate task IDs",
    ):
        runner.run((task, task))

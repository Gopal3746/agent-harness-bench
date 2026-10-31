from pathlib import Path

import pytest

from agent_harness import CommandRunner, WorkspaceManager
from agent_harness.cli import load_task

_TASK_ROOT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "tasks"
)

_BENCHMARK_CASES = (
    (
        "audit-events",
        "audit-events",
        (
            "observability",
            "multi-module",
            "decomposable",
        ),
        "9 failed, 3 passed",
    ),
    (
        "config-merge",
        "config-merge",
        (
            "configuration",
            "recursive",
            "tightly-coupled",
        ),
        "3 failed, 3 passed",
    ),
    (
        "input-validation",
        "input-validation",
        (
            "validation",
            "multi-module",
            "decomposable",
        ),
        "5 failed, 4 passed",
    ),
    (
        "pagination-client",
        "pagination-client",
        (
            "api-client",
            "single-module",
            "non-decomposable",
        ),
        "5 failed, 1 passed",
    ),
    (
        "service-retries",
        "service-retries",
        (
            "retries",
            "multi-module",
            "decomposable",
        ),
        "9 failed, 6 passed",
    ),
    (
        "ttl-cache",
        "ttl-cache-expiration",
        (
            "cache",
            "stateful",
            "tightly-coupled",
        ),
        "5 failed, 4 passed",
    ),
)


@pytest.mark.parametrize(
    (
        "directory",
        "task_id",
        "tags",
        "expected_summary",
    ),
    _BENCHMARK_CASES,
)
def test_task_has_reproducible_failing_baseline(
    tmp_path: Path,
    directory: str,
    task_id: str,
    tags: tuple[str, ...],
    expected_summary: str,
) -> None:
    task = load_task(
        _TASK_ROOT
        / directory
        / "task.json"
    )
    workspaces = WorkspaceManager(
        tmp_path / task_id
    )

    with workspaces.provision(
        task,
        run_id=f"{task_id}-baseline",
    ) as workspace:
        result = CommandRunner(
            workspace.path
        ).run_tests(task)

    assert task.task_id == task_id
    assert task.tags == tags
    assert result.succeeded is False
    assert result.exit_code == 1
    assert expected_summary in result.stdout

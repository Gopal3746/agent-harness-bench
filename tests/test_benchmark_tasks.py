from pathlib import Path

from agent_harness import CommandRunner, WorkspaceManager
from agent_harness.cli import load_task

_TASK_ROOT = (
    Path(__file__).parent.parent
    / "benchmarks"
    / "tasks"
    / "input-validation"
)


def test_input_validation_task_has_reproducible_failing_baseline(
    tmp_path: Path,
) -> None:
    task = load_task(_TASK_ROOT / "task.json")
    workspaces = WorkspaceManager(tmp_path / "workspaces")

    with workspaces.provision(
        task,
        run_id="baseline",
    ) as workspace:
        result = CommandRunner(workspace.path).run_tests(task)

    assert task.task_id == "input-validation"
    assert task.tags == (
        "validation",
        "multi-module",
        "decomposable",
    )
    assert result.succeeded is False
    assert result.exit_code == 1
    assert "5 failed, 4 passed" in result.stdout

from pathlib import Path

import pytest

from agent_harness import (
    TaskSpec,
    TaskWorkspace,
    WorkspaceManager,
)


def make_task(tmp_path: Path) -> TaskSpec:
    source = tmp_path / "task-source"
    package = source / "package"
    package.mkdir(parents=True)

    (package / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    git_directory = source / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text(
        "generated metadata\n",
        encoding="utf-8",
    )

    cache_directory = package / "__pycache__"
    cache_directory.mkdir()
    (cache_directory / "app.pyc").write_bytes(b"generated")

    return TaskSpec(
        task_id="fix-parser",
        instruction="Fix the parser tests.",
        source_dir=source,
    )


def test_create_copies_task_into_isolated_workspace(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(tmp_path / "workspaces")

    workspace = manager.create(task, run_id="run-001")

    assert workspace.task_id == "fix-parser"
    assert workspace.run_id == "run-001"
    assert workspace.path != task.source_dir
    assert (workspace.path / "package" / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"

    assert not (workspace.path / ".git").exists()
    assert not (
        workspace.path / "package" / "__pycache__"
    ).exists()


def test_create_generates_distinct_run_ids(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(tmp_path / "workspaces")

    first = manager.create(task)
    second = manager.create(task)

    assert first.run_id != second.run_id
    assert first.path != second.path


def test_create_rejects_missing_source(
    tmp_path: Path,
) -> None:
    task = TaskSpec(
        task_id="missing-task",
        instruction="Fix the missing task.",
        source_dir=tmp_path / "does-not-exist",
    )
    manager = WorkspaceManager(tmp_path / "workspaces")

    with pytest.raises(FileNotFoundError):
        manager.create(task)


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "contains spaces",
        "/absolute",
    ],
)
def test_create_rejects_unsafe_run_id(
    tmp_path: Path,
    run_id: str,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(tmp_path / "workspaces")

    with pytest.raises(
        ValueError,
        match="run_id must contain only",
    ):
        manager.create(task, run_id=run_id)


def test_create_rejects_duplicate_workspace(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(tmp_path / "workspaces")

    manager.create(task, run_id="same-run")

    with pytest.raises(
        FileExistsError,
        match="workspace already exists",
    ):
        manager.create(task, run_id="same-run")


def test_provision_removes_workspace_after_use(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(tmp_path / "workspaces")

    with manager.provision(
        task,
        run_id="temporary-run",
    ) as workspace:
        workspace_path = workspace.path
        assert workspace_path.exists()

    assert not workspace_path.exists()


def test_provision_can_retain_workspace(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    manager = WorkspaceManager(
        tmp_path / "workspaces",
        retain_workspaces=True,
    )

    with manager.provision(
        task,
        run_id="retained-run",
    ) as workspace:
        workspace_path = workspace.path

    assert workspace_path.exists()


def test_remove_rejects_path_outside_workspace_root(
    tmp_path: Path,
) -> None:
    manager = WorkspaceManager(tmp_path / "workspaces")
    outside = tmp_path / "outside"
    outside.mkdir()

    workspace = TaskWorkspace(
        task_id="unsafe",
        run_id="unsafe",
        path=outside,
    )

    with pytest.raises(
        ValueError,
        match="outside workspace root",
    ):
        manager.remove(workspace)

    assert outside.exists()

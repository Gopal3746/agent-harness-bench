from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_harness import discover_tasks


def write_task(
    root: Path,
    directory: str,
    task_id: str,
) -> Path:
    task_directory = root / directory
    repository = task_directory / "repository"
    repository.mkdir(parents=True)

    manifest = task_directory / "task.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "instruction": "Complete the task.",
                "source_dir": "repository",
                "test_command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                ],
                "timeout_seconds": 30,
                "tags": ["catalog"],
            }
        ),
        encoding="utf-8",
    )

    return repository


def test_discover_tasks_returns_sorted_validated_tasks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tasks"
    second_repository = write_task(
        root,
        "second",
        "task-b",
    )
    first_repository = write_task(
        root,
        "first",
        "task-a",
    )

    tasks = discover_tasks(root)

    assert [
        task.task_id
        for task in tasks
    ] == ["task-a", "task-b"]

    assert tasks[0].source_dir == (
        first_repository.resolve()
    )
    assert tasks[1].source_dir == (
        second_repository.resolve()
    )


def test_discover_tasks_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tasks"
    write_task(root, "first", "duplicate")
    write_task(root, "second", "duplicate")

    with pytest.raises(
        ValueError,
        match="duplicate task_id 'duplicate'",
    ):
        discover_tasks(root)


def test_discover_tasks_rejects_empty_catalog(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tasks"
    root.mkdir()

    with pytest.raises(
        ValueError,
        match="no task manifests",
    ):
        discover_tasks(root)

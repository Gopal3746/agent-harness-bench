from pathlib import Path

import pytest

from agent_harness import RepositoryTools, SubtaskSpec, TaskPlan, TaskSpec
from agent_harness.agent.worker import (
    build_worker_task,
    load_file_context,
)


def test_file_context_is_shared_bounded_and_subtask_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "users.py").write_text(
        "USER_LIMIT = 1\n" + "u" * 100,
        encoding="utf-8",
    )
    (tmp_path / "network.py").write_text(
        "PORT_LIMIT = 1\n" + "n" * 100,
        encoding="utf-8",
    )
    repository = RepositoryTools(tmp_path)
    original_read_file = repository.read_file
    reads: list[str] = []

    def count_read(path: str) -> str:
        reads.append(path)
        return original_read_file(path)

    monkeypatch.setattr(repository, "read_file", count_read)

    users = SubtaskSpec(
        subtask_id="users",
        instruction="Update the user limit.",
        relevant_files=("users.py",),
    )
    network = SubtaskSpec(
        subtask_id="network",
        instruction="Update the network limit.",
        relevant_files=("users.py", "network.py"),
    )
    plan = TaskPlan(subtasks=(users, network))

    context = load_file_context(
        repository,
        plan.subtasks,
        max_total_chars=110,
        max_file_chars=60,
    )

    assert reads == ["users.py", "network.py"]
    assert len(context["users.py"]) <= 60
    assert sum(len(value) for value in context.values()) <= 110

    task = TaskSpec(
        task_id="limits",
        instruction="Update independent limits.",
        source_dir=tmp_path,
    )
    worker_task = build_worker_task(
        task,
        users,
        plan,
        isolated_workspace=True,
        file_context=context,
    )

    assert "--- users.py ---" in worker_task.instruction
    assert "USER_LIMIT = 1" in worker_task.instruction
    assert "--- network.py ---" not in worker_task.instruction
    assert "file context truncated" in worker_task.instruction

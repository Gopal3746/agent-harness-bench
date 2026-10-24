from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_harness.models import TaskSpec


def load_task(path: Path) -> TaskSpec:
    """Load a task and resolve its source directory."""

    resolved_path = path.expanduser().resolve()
    payload: Any = json.loads(
        resolved_path.read_text(encoding="utf-8")
    )

    if isinstance(payload, dict):
        source_dir = payload.get("source_dir")

        if isinstance(source_dir, str):
            source_path = Path(source_dir).expanduser()

            if not source_path.is_absolute():
                source_path = (
                    resolved_path.parent / source_path
                )

            payload["source_dir"] = source_path.resolve()

    return TaskSpec.model_validate(payload)


def discover_tasks(root: Path) -> tuple[TaskSpec, ...]:
    """Discover and validate task manifests recursively."""

    resolved_root = root.expanduser().resolve()

    if not resolved_root.is_dir():
        raise FileNotFoundError(
            f"task catalog does not exist: {resolved_root}"
        )

    manifests = sorted(
        resolved_root.rglob("task.json")
    )

    if not manifests:
        raise ValueError(
            f"no task manifests found under {resolved_root}"
        )

    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()

    for manifest in manifests:
        task = load_task(manifest)

        if task.task_id in seen_ids:
            raise ValueError(
                f"duplicate task_id {task.task_id!r}"
            )

        seen_ids.add(task.task_id)
        tasks.append(task)

    return tuple(
        sorted(
            tasks,
            key=lambda task: task.task_id,
        )
    )

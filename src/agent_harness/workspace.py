from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent_harness.models import TaskSpec

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.pyc",
)


@dataclass(frozen=True, slots=True)
class TaskWorkspace:
    """An isolated copy of a coding task for one benchmark run."""

    task_id: str
    run_id: str
    path: Path


class WorkspaceManager:
    """Creates and cleans isolated task workspaces."""

    def __init__(
        self,
        root: Path,
        *,
        retain_workspaces: bool = False,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.retain_workspaces = retain_workspaces

    def create(
        self,
        task: TaskSpec,
        *,
        run_id: str | None = None,
    ) -> TaskWorkspace:
        """Copy a task's source directory into a fresh workspace."""

        source = task.source_dir.expanduser().resolve()

        if not source.is_dir():
            raise FileNotFoundError(
                f"task source directory does not exist: {source}"
            )

        if self.root.is_relative_to(source) or source.is_relative_to(
            self.root
        ):
            raise ValueError(
                "workspace root and task source directory must not overlap"
            )

        resolved_run_id = run_id or uuid4().hex
        self._validate_run_id(resolved_run_id)

        destination = self.root / task.task_id / resolved_run_id

        if destination.exists():
            raise FileExistsError(
                f"workspace already exists: {destination}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            ignore=_IGNORE_PATTERNS,
        )

        return TaskWorkspace(
            task_id=task.task_id,
            run_id=resolved_run_id,
            path=destination,
        )

    def remove(self, workspace: TaskWorkspace) -> None:
        """Remove a workspace after verifying it belongs to this manager."""

        destination = workspace.path.expanduser().resolve()

        if destination == self.root or not destination.is_relative_to(
            self.root
        ):
            raise ValueError(
                f"refusing to remove path outside workspace root: "
                f"{destination}"
            )

        if destination.exists():
            shutil.rmtree(destination)

        task_directory = destination.parent
        if (
            task_directory != self.root
            and task_directory.exists()
            and not any(task_directory.iterdir())
        ):
            task_directory.rmdir()

    @contextmanager
    def provision(
        self,
        task: TaskSpec,
        *,
        run_id: str | None = None,
    ) -> Iterator[TaskWorkspace]:
        """Create a workspace and clean it after the run."""

        workspace = self.create(task, run_id=run_id)

        try:
            yield workspace
        finally:
            if not self.retain_workspaces:
                self.remove(workspace)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, "
                "underscores, periods, and hyphens"
            )

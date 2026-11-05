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


@dataclass(frozen=True, slots=True)
class _ProtectedFile:
    """Contents and permissions for one protected verifier file."""

    content: bytes
    mode: int


@dataclass(frozen=True, slots=True)
class VerifierSnapshot:
    """Immutable copy of verifier-owned paths in a workspace."""

    root: Path
    protected_paths: tuple[Path, ...]
    directories: tuple[tuple[Path, int], ...]
    files: tuple[tuple[Path, _ProtectedFile], ...]

    @classmethod
    def capture(
        cls,
        root: Path,
        protected_paths: tuple[str, ...],
    ) -> VerifierSnapshot:
        """Capture protected paths before the agent starts."""

        resolved_root = root.expanduser().resolve()
        normalized = cls._normalize_protected_paths(
            protected_paths
        )
        directories: list[tuple[Path, int]] = []
        files: list[tuple[Path, _ProtectedFile]] = []

        for protected_path in normalized:
            target = resolved_root / protected_path

            if target.is_symlink():
                raise ValueError(
                    "protected verifier paths must not be symlinks: "
                    f"{protected_path.as_posix()}"
                )

            if not target.exists():
                continue

            candidates = (
                (target,)
                if target.is_file()
                else (target, *sorted(target.rglob("*")))
            )

            for candidate in candidates:
                relative = candidate.relative_to(resolved_root)

                if candidate.is_symlink():
                    raise ValueError(
                        "protected verifier paths must not contain "
                        f"symlinks: {relative.as_posix()}"
                    )

                mode = candidate.stat().st_mode & 0o7777

                if candidate.is_dir():
                    directories.append((relative, mode))
                elif candidate.is_file():
                    files.append(
                        (
                            relative,
                            _ProtectedFile(
                                content=candidate.read_bytes(),
                                mode=mode,
                            ),
                        )
                    )
                else:
                    raise ValueError(
                        "protected verifier paths must contain only "
                        f"files and directories: {relative.as_posix()}"
                    )

        return cls(
            root=resolved_root,
            protected_paths=normalized,
            directories=tuple(directories),
            files=tuple(files),
        )

    def restore(self) -> None:
        """Restore the exact protected state before verification."""

        for protected_path in self.protected_paths:
            self._remove_path(self.root / protected_path)

        for relative, _ in sorted(
            self.directories,
            key=lambda item: len(item[0].parts),
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

        for relative, protected_file in self.files:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(protected_file.content)
            target.chmod(protected_file.mode)

        for relative, mode in sorted(
            self.directories,
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            (self.root / relative).chmod(mode)

    @staticmethod
    def _normalize_protected_paths(
        protected_paths: tuple[str, ...],
    ) -> tuple[Path, ...]:
        normalized: list[Path] = []

        for supplied in protected_paths:
            cleaned = supplied.strip().rstrip("/")
            path = Path(cleaned)

            if (
                not cleaned
                or path.is_absolute()
                or len(path.parts) != 1
                or path.parts[0] in {".", ".."}
            ):
                raise ValueError(
                    "protected paths must be top-level relative paths"
                )

            normalized.append(path)

        return tuple(normalized)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


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

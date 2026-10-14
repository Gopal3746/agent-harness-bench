from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

_IGNORED_NAMES = frozenset(
    {
        ".agent-runtime",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)


class WorkspaceMergeError(RuntimeError):
    """Isolated subagent changes cannot be merged safely."""


@dataclass(frozen=True, slots=True)
class FileChange:
    """The desired content of one created, updated, or deleted file."""

    path: str
    content: bytes | None

    @property
    def deleted(self) -> bool:
        return self.content is None


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """Immutable file-content snapshot of a repository workspace."""

    files: tuple[tuple[str, bytes], ...]

    @classmethod
    def capture(cls, root: Path) -> WorkspaceSnapshot:
        """Capture non-runtime files beneath an existing directory."""

        resolved_root = root.expanduser().resolve()

        if not resolved_root.is_dir():
            raise FileNotFoundError(
                f"workspace root does not exist: {resolved_root}"
            )

        files: list[tuple[str, bytes]] = []

        for path in resolved_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue

            relative = path.relative_to(resolved_root)

            if any(part in _IGNORED_NAMES for part in relative.parts):
                continue

            files.append(
                (relative.as_posix(), path.read_bytes())
            )

        return cls(files=tuple(sorted(files)))

    def changes_since(
        self,
        baseline: WorkspaceSnapshot,
    ) -> tuple[FileChange, ...]:
        """Return changes needed to transform baseline into this snapshot."""

        before = dict(baseline.files)
        after = dict(self.files)

        return tuple(
            FileChange(
                path=path,
                content=after.get(path),
            )
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)
        )


@dataclass(frozen=True, slots=True)
class WorkspaceChangeSet:
    """File changes produced by one isolated subagent."""

    owner: str
    changes: tuple[FileChange, ...]

    @classmethod
    def between(
        cls,
        owner: str,
        baseline: WorkspaceSnapshot,
        updated: WorkspaceSnapshot,
    ) -> WorkspaceChangeSet:
        cleaned_owner = owner.strip()

        if not cleaned_owner:
            raise ValueError("change-set owner must not be blank")

        return cls(
            owner=cleaned_owner,
            changes=updated.changes_since(baseline),
        )


def clone_workspace(source: Path, destination: Path) -> Path:
    """Copy a workspace while excluding harness and tool caches."""

    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()

    if not resolved_source.is_dir():
        raise FileNotFoundError(
            f"workspace root does not exist: {resolved_source}"
        )

    if resolved_destination.exists():
        raise FileExistsError(
            f"clone destination already exists: {resolved_destination}"
        )

    shutil.copytree(
        resolved_source,
        resolved_destination,
        ignore=shutil.ignore_patterns(*_IGNORED_NAMES),
    )

    return resolved_destination


def merge_change_sets(
    root: Path,
    change_sets: tuple[WorkspaceChangeSet, ...],
) -> tuple[str, ...]:
    """Merge compatible changes and reject conflicts before writing."""

    resolved_root = root.expanduser().resolve()

    if not resolved_root.is_dir():
        raise FileNotFoundError(
            f"workspace root does not exist: {resolved_root}"
        )

    merged: dict[str, bytes | None] = {}
    owners: dict[str, str] = {}

    for change_set in change_sets:
        for change in change_set.changes:
            if (
                change.path in merged
                and merged[change.path] != change.content
            ):
                raise WorkspaceMergeError(
                    f"conflicting changes to {change.path!r} from "
                    f"{owners[change.path]!r} and {change_set.owner!r}"
                )

            merged[change.path] = change.content
            owners[change.path] = change_set.owner

    targets: dict[str, Path] = {}

    for relative_path in merged:
        target = (resolved_root / relative_path).resolve()

        if not target.is_relative_to(resolved_root):
            raise WorkspaceMergeError(
                f"change path escapes workspace: {relative_path!r}"
            )

        targets[relative_path] = target

    for relative_path, content in sorted(merged.items()):
        target = targets[relative_path]

        if content is None:
            target.unlink(missing_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    return tuple(sorted(merged))

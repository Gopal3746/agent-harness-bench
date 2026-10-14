from pathlib import Path

import pytest

from agent_harness import (
    FileChange,
    WorkspaceChangeSet,
    WorkspaceMergeError,
    WorkspaceSnapshot,
    clone_workspace,
    merge_change_sets,
)


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    (workspace / "users.py").write_text(
        "USER_LIMIT = 1\n",
        encoding="utf-8",
    )
    (workspace / "network.py").write_text(
        "PORT_LIMIT = 1\n",
        encoding="utf-8",
    )

    return workspace


def test_snapshot_detects_created_updated_and_deleted_files(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    baseline = WorkspaceSnapshot.capture(workspace)

    (workspace / "users.py").write_text(
        "USER_LIMIT = 2\n",
        encoding="utf-8",
    )
    (workspace / "network.py").unlink()
    (workspace / "new.py").write_text(
        "ENABLED = True\n",
        encoding="utf-8",
    )

    updated = WorkspaceSnapshot.capture(workspace)
    changes = updated.changes_since(baseline)

    assert [change.path for change in changes] == [
        "network.py",
        "new.py",
        "users.py",
    ]
    assert changes[0].deleted is True
    assert changes[1].content == b"ENABLED = True\n"
    assert changes[2].content == b"USER_LIMIT = 2\n"


def test_clone_excludes_runtime_directories(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    runtime = workspace / ".agent-runtime"
    runtime.mkdir()

    (runtime / "state.json").write_text(
        "{}",
        encoding="utf-8",
    )

    clone = clone_workspace(
        workspace,
        tmp_path / "clone",
    )

    assert (clone / "users.py").is_file()
    assert not (clone / ".agent-runtime").exists()


def test_merge_combines_independent_change_sets(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    baseline = WorkspaceSnapshot.capture(workspace)

    users_clone = clone_workspace(
        workspace,
        tmp_path / "users-clone",
    )
    (users_clone / "users.py").write_text(
        "USER_LIMIT = 2\n",
        encoding="utf-8",
    )

    network_clone = clone_workspace(
        workspace,
        tmp_path / "network-clone",
    )
    (network_clone / "network.py").write_text(
        "PORT_LIMIT = 2\n",
        encoding="utf-8",
    )

    changed_paths = merge_change_sets(
        workspace,
        (
            WorkspaceChangeSet.between(
                "users",
                baseline,
                WorkspaceSnapshot.capture(users_clone),
            ),
            WorkspaceChangeSet.between(
                "network",
                baseline,
                WorkspaceSnapshot.capture(network_clone),
            ),
        ),
    )

    assert changed_paths == ("network.py", "users.py")

    assert (workspace / "users.py").read_text(
        encoding="utf-8"
    ) == "USER_LIMIT = 2\n"

    assert (workspace / "network.py").read_text(
        encoding="utf-8"
    ) == "PORT_LIMIT = 2\n"


def test_merge_rejects_conflicts_before_writing(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    baseline = WorkspaceSnapshot.capture(workspace)

    first_clone = clone_workspace(
        workspace,
        tmp_path / "first-clone",
    )
    (first_clone / "users.py").write_text(
        "USER_LIMIT = 2\n",
        encoding="utf-8",
    )
    (first_clone / "new.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    second_clone = clone_workspace(
        workspace,
        tmp_path / "second-clone",
    )
    (second_clone / "users.py").write_text(
        "USER_LIMIT = 3\n",
        encoding="utf-8",
    )

    first = WorkspaceChangeSet.between(
        "first",
        baseline,
        WorkspaceSnapshot.capture(first_clone),
    )
    second = WorkspaceChangeSet.between(
        "second",
        baseline,
        WorkspaceSnapshot.capture(second_clone),
    )

    with pytest.raises(
        WorkspaceMergeError,
        match="conflicting changes to 'users.py'",
    ):
        merge_change_sets(workspace, (first, second))

    assert (workspace / "users.py").read_text(
        encoding="utf-8"
    ) == "USER_LIMIT = 1\n"

    assert not (workspace / "new.py").exists()


def test_merge_rejects_unsafe_paths_before_writing(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    changes = WorkspaceChangeSet(
        owner="unsafe",
        changes=(
            FileChange("new.py", b"VALUE = 1\n"),
            FileChange("z/../../escape.py", b"unsafe\n"),
        ),
    )

    with pytest.raises(
        WorkspaceMergeError,
        match="escapes workspace",
    ):
        merge_change_sets(workspace, (changes,))

    assert not (workspace / "new.py").exists()
    assert not (tmp_path / "escape.py").exists()

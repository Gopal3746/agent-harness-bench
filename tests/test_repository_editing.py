import stat
from pathlib import Path

import pytest

from agent_harness import (
    RepositoryTools,
    ToolExecutionError,
)


def make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "app.py").write_text(
        "VALUE = 1\nVALUE = 1\n",
        encoding="utf-8",
    )

    return repository


def test_write_file_creates_nested_file(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    result = tools.write_file(
        "src/generated.py",
        "ENABLED = True\n",
    )

    assert result.path == "src/generated.py"
    assert result.created is True
    assert result.bytes_written == len(b"ENABLED = True\n")
    assert (repository / "src" / "generated.py").read_text(
        encoding="utf-8"
    ) == "ENABLED = True\n"


def test_write_file_can_refuse_overwrite(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="file already exists",
    ):
        tools.write_file(
            "app.py",
            "VALUE = 2\n",
            overwrite=False,
        )

    assert (repository / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\nVALUE = 1\n"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/tmp/outside.py",
    ],
)
def test_write_file_rejects_unsafe_paths(
    tmp_path: Path,
    path: str,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(ToolExecutionError):
        tools.write_file(path, "unsafe = True\n")


def test_write_file_rejects_ignored_directory(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="ignored repository directory",
    ):
        tools.write_file(".git/config", "unsafe\n")


def test_write_file_enforces_size_limit(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(
        repository,
        max_write_bytes=5,
    )

    with pytest.raises(
        ToolExecutionError,
        match="content exceeds",
    ):
        tools.write_file("large.txt", "123456")


def test_replace_text_replaces_exact_occurrences(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    result = tools.replace_text(
        "app.py",
        "VALUE = 1",
        "VALUE = 2",
        expected_replacements=2,
    )

    assert result.created is False
    assert (repository / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\nVALUE = 2\n"


def test_replace_text_rejects_unexpected_count(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)
    original = (repository / "app.py").read_text(
        encoding="utf-8"
    )

    with pytest.raises(
        ToolExecutionError,
        match="expected 1 occurrences, found 2",
    ):
        tools.replace_text(
            "app.py",
            "VALUE = 1",
            "VALUE = 2",
        )

    assert (repository / "app.py").read_text(
        encoding="utf-8"
    ) == original


def test_replace_text_rejects_empty_search(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="must not be empty",
    ):
        tools.replace_text("app.py", "", "replacement")


def test_write_file_preserves_existing_permissions(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    target = repository / "script.py"
    target.write_text(
        "#!/usr/bin/env python3\n",
        encoding="utf-8",
    )
    target.chmod(0o755)

    tools = RepositoryTools(repository)
    tools.write_file(
        "script.py",
        "#!/usr/bin/env python3\nprint('ready')\n",
    )

    assert stat.S_IMODE(target.stat().st_mode) == 0o755

from pathlib import Path

import pytest

from agent_harness import (
    RepositoryTools,
    ToolExecutionError,
)


def make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    source = repository / "src"
    source.mkdir(parents=True)

    (repository / "README.md").write_text(
        "# Example\n",
        encoding="utf-8",
    )
    (source / "app.py").write_text(
        "def retry_request():\n"
        "    return 'retry enabled'\n",
        encoding="utf-8",
    )
    (source / "client.py").write_text(
        "FIRST = 1\n"
        "SECOND = 2\n"
        "THIRD = 3\n",
        encoding="utf-8",
    )

    git_directory = repository / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text(
        "secret metadata\n",
        encoding="utf-8",
    )

    return repository


def test_list_files_returns_sorted_repository_paths(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    assert tools.list_files() == (
        "README.md",
        "src/app.py",
        "src/client.py",
    )


def test_read_file_returns_requested_line_range(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    content = tools.read_file(
        "src/client.py",
        start_line=2,
        end_line=3,
    )

    assert content == "SECOND = 2\nTHIRD = 3\n"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/tmp/outside.txt",
    ],
)
def test_read_file_rejects_unsafe_paths(
    tmp_path: Path,
    path: str,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(ToolExecutionError):
        tools.read_file(path)


def test_read_file_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private\n", encoding="utf-8")

    link = repository / "outside-link.txt"
    link.symlink_to(outside)

    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="escapes the repository",
    ):
        tools.read_file("outside-link.txt")


def test_read_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="file does not exist",
    ):
        tools.read_file("missing.py")


def test_search_repo_returns_paths_and_line_numbers(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    matches = tools.search_repo("RETRY")

    assert len(matches) == 2

    assert matches[0].path == "src/app.py"
    assert matches[0].line_number == 1
    assert matches[0].text == "def retry_request():"

    assert matches[1].path == "src/app.py"
    assert matches[1].line_number == 2
    assert matches[1].text == "    return 'retry enabled'"


def test_search_repo_rejects_empty_query(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    with pytest.raises(
        ToolExecutionError,
        match="query must not be empty",
    ):
        tools.search_repo("")


def test_repository_tools_requires_existing_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        RepositoryTools(tmp_path / "missing")

def test_search_repo_respects_result_limit(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tools = RepositoryTools(repository)

    matches = tools.search_repo(
        "=",
        path="src",
        max_results=2,
    )

    assert len(matches) == 2

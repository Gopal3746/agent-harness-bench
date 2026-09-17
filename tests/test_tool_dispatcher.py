import json
from pathlib import Path

from agent_harness import (
    CommandRunner,
    RepositoryTools,
    TaskSpec,
    ToolDispatcher,
    build_tool_schemas,
)


def make_dispatcher(
    tmp_path: Path,
) -> tuple[ToolDispatcher, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "app.py").write_text(
        "def greet():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    task = TaskSpec(
        task_id="greeting-task",
        instruction="Update the greeting.",
        source_dir=repository,
        test_command=(
            "python",
            "-c",
            "print('verification passed')",
        ),
        timeout_seconds=30,
    )

    dispatcher = ToolDispatcher(
        repository=RepositoryTools(repository),
        commands=CommandRunner(repository),
        task=task,
    )

    return dispatcher, repository


def test_tool_schemas_are_openai_compatible() -> None:
    schemas = build_tool_schemas()

    names = {
        schema["function"]["name"]
        for schema in schemas
    }

    assert names == {
        "list_files",
        "read_file",
        "replace_text",
        "run_command",
        "run_tests",
        "search_repo",
        "write_file",
    }

    for schema in schemas:
        assert schema["type"] == "function"
        parameters = schema["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False


def test_dispatch_lists_repository_files(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "list_files",
        '{"path": "."}',
    )

    assert response.is_error is False
    assert json.loads(response.content) == ["app.py"]
    assert response.duration_ms >= 0


def test_dispatch_searches_repository(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "search_repo",
        '{"query": "greet"}',
    )

    results = json.loads(response.content)

    assert response.is_error is False
    assert results[0]["path"] == "app.py"
    assert results[0]["line_number"] == 1


def test_dispatch_writes_and_replaces_text(
    tmp_path: Path,
) -> None:
    dispatcher, repository = make_dispatcher(tmp_path)

    write_response = dispatcher.dispatch(
        "write_file",
        json.dumps(
            {
                "path": "new_file.py",
                "content": "VALUE = 1\n",
            }
        ),
    )

    replace_response = dispatcher.dispatch(
        "replace_text",
        json.dumps(
            {
                "path": "new_file.py",
                "old": "VALUE = 1",
                "new": "VALUE = 2",
            }
        ),
    )

    assert write_response.is_error is False
    assert replace_response.is_error is False
    assert (repository / "new_file.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"


def test_dispatch_runs_command(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [
                    "python",
                    "-c",
                    "print('ready')",
                ]
            }
        ),
    )

    result = json.loads(response.content)

    assert response.is_error is False
    assert result["succeeded"] is True
    assert result["stdout"] == "ready\n"
    assert result["exit_code"] == 0


def test_dispatch_runs_task_tests(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch("run_tests", "{}")
    result = json.loads(response.content)

    assert response.is_error is False
    assert result["succeeded"] is True
    assert result["stdout"] == "verification passed\n"


def test_dispatch_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "list_files",
        "{not valid json",
    )

    assert response.is_error is True
    assert "invalid tool arguments" in response.content


def test_dispatch_rejects_invalid_arguments(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "read_file",
        '{"path": "app.py", "unexpected": true}',
    )

    assert response.is_error is True
    assert "invalid tool arguments" in response.content


def test_dispatch_rejects_unknown_tool(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "delete_everything",
        "{}",
    )

    assert response.is_error is True
    assert json.loads(response.content) == {
        "error": "unknown tool: delete_everything"
    }


def test_dispatch_returns_tool_execution_error(
    tmp_path: Path,
) -> None:
    dispatcher, _ = make_dispatcher(tmp_path)

    response = dispatcher.dispatch(
        "read_file",
        '{"path": "../outside.txt"}',
    )

    assert response.is_error is True
    assert "escapes the repository workspace" in response.content

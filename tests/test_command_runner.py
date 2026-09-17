from pathlib import Path

import pytest

from agent_harness import (
    CommandRunner,
    TaskSpec,
    ToolExecutionError,
)


def make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    return repository


def test_run_captures_stdout_stderr_and_duration(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(repository)

    result = runner.run(
        (
            "python",
            "-c",
            (
                "import sys; "
                "print('standard output'); "
                "print('standard error', file=sys.stderr)"
            ),
        )
    )

    assert result.succeeded is True
    assert result.exit_code == 0
    assert result.stdout == "standard output\n"
    assert result.stderr == "standard error\n"
    assert result.duration_ms > 0
    assert result.timed_out is False
    assert result.output_truncated is False


def test_run_captures_nonzero_exit_code(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(repository)

    result = runner.run(
        (
            "python",
            "-c",
            "raise SystemExit(7)",
        )
    )

    assert result.succeeded is False
    assert result.exit_code == 7
    assert result.timed_out is False


def test_run_rejects_disallowed_command(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(repository)

    with pytest.raises(
        ToolExecutionError,
        match="command is not allowed",
    ):
        runner.run(("sh", "-c", "echo unsafe"))


def test_run_rejects_executable_path(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(repository)

    with pytest.raises(
        ToolExecutionError,
        match="executable paths are not allowed",
    ):
        runner.run(("/bin/sh", "-c", "echo unsafe"))


def test_run_rejects_working_directory_escape(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(repository)

    with pytest.raises(
        ToolExecutionError,
        match="escapes the repository",
    ):
        runner.run(
            ("python", "-c", "print('unsafe')"),
            cwd="../",
        )


def test_run_terminates_command_after_timeout(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(
        repository,
        default_timeout_seconds=1,
    )

    result = runner.run(
        (
            "python",
            "-c",
            "import time; time.sleep(5)",
        ),
        timeout_seconds=0.05,
    )

    assert result.succeeded is False
    assert result.timed_out is True
    assert result.exit_code is None
    assert result.duration_ms < 2_000


def test_run_truncates_large_output(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(
        repository,
        max_output_chars=100,
    )

    result = runner.run(
        (
            "python",
            "-c",
            "print('a' * 500)",
        )
    )

    assert result.succeeded is True
    assert result.output_truncated is True
    assert "output truncated" in result.stdout
    assert len(result.stdout) <= 100


def test_run_tests_uses_task_configuration(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    tests_directory = repository / "tests"
    tests_directory.mkdir()

    (tests_directory / "test_sample.py").write_text(
        "def test_sample():\n"
        "    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    task = TaskSpec(
        task_id="sample-task",
        instruction="Run the sample test.",
        source_dir=repository,
        test_command=("python", "-m", "pytest", "-q"),
        timeout_seconds=30,
    )

    runner = CommandRunner(repository)
    result = runner.run_tests(task)

    assert result.succeeded is True
    assert "1 passed" in result.stdout


def test_run_rejects_timeout_above_maximum(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    runner = CommandRunner(
        repository,
        default_timeout_seconds=1,
        max_timeout_seconds=2,
    )

    with pytest.raises(
        ToolExecutionError,
        match="exceeds maximum",
    ):
        runner.run(
            ("python", "-c", "print('ready')"),
            timeout_seconds=3,
        )

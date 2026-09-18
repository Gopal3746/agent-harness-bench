from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest

from agent_harness import ChatCompletionResult, cli


class FakeOpenRouterClient:
    api_keys: ClassVar[list[str]] = []

    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.api_keys.append(api_key)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        session_id: str | None = None,
    ) -> ChatCompletionResult:
        del messages, tools, temperature, max_tokens, seed, session_id

        return ChatCompletionResult(
            request_id="request-id",
            generation_id="generation-id",
            model=model,
            content="The task is complete.",
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
            usage=None,
            request_started_ns=1_000_000_000,
            first_output_ns=1_050_000_000,
            last_output_ns=1_100_000_000,
            request_finished_ns=1_150_000_000,
            stream_event_timestamps_ns=(
                1_050_000_000,
                1_100_000_000,
            ),
            raw_event_count=2,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


def write_task(tmp_path: Path, *, passing: bool = True) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    manifest = tmp_path / "task.json"
    expected_value = 1 if passing else 2
    manifest.write_text(
        json.dumps(
            {
                "task_id": "sample-task",
                "instruction": "Ensure VALUE is correct.",
                "source_dir": "repository",
                "test_command": [
                    "python",
                    "-c",
                    f"import app; assert app.VALUE == {expected_value}",
                ],
                "timeout_seconds": 30,
                "tags": ["cli"],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_cli_runs_without_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 0
    assert "run and verify one coding task" in capsys.readouterr().out


def test_cli_reports_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "agent-harness 0.1.0"


def test_load_task_resolves_relative_source_directory(
    tmp_path: Path,
) -> None:
    manifest = write_task(tmp_path)

    task = cli.load_task(manifest)

    assert task.source_dir == (tmp_path / "repository").resolve()


def test_run_command_writes_passing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = write_task(tmp_path)
    output = tmp_path / "results" / "runs.jsonl"
    workspaces = tmp_path / "workspaces"
    FakeOpenRouterClient.api_keys.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(
        cli,
        "OpenRouterClient",
        FakeOpenRouterClient,
    )

    exit_code = cli.main(
        [
            "run",
            str(manifest),
            "--model",
            "test/model",
            "--output",
            str(output),
            "--workspace-root",
            str(workspaces),
            "--run-id",
            "cli-run-001",
        ]
    )

    assert exit_code == 0
    assert FakeOpenRouterClient.api_keys == ["test-secret"]

    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is True
    assert summary["run_id"] == "cli-run-001"

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["task_id"] == "sample-task"
    assert record["passed"] is True
    assert list(workspaces.iterdir()) == []


def test_run_command_returns_one_when_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_task(tmp_path, passing=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(
        cli,
        "OpenRouterClient",
        FakeOpenRouterClient,
    )

    exit_code = cli.main(
        [
            "run",
            str(manifest),
            "--model",
            "test/model",
            "--output",
            str(tmp_path / "runs.jsonl"),
            "--workspace-root",
            str(tmp_path / "workspaces"),
        ]
    )

    assert exit_code == 1


def test_run_command_requires_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_task(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "run",
                str(manifest),
                "--model",
                "test/model",
            ]
        )

    assert exc_info.value.code == 2

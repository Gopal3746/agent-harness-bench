from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_harness import (
    BenchmarkConfig,
    BenchmarkRunner,
    ChatCompletionResult,
    ExecutionStrategy,
    JsonlRunWriter,
    TaskSpec,
    WorkspaceManager,
)


class FakeChatClient:
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
        del tools, temperature, max_tokens, seed, session_id

        content = "Task complete."

        if messages and "decompose repository-level" in str(
            messages[0].get("content", "")
        ):
            content = (
                '{"subtasks": [{"subtask_id": "inspect", '
                '"instruction": "Inspect the repository.", '
                '"depends_on": []}]}'
            )

        return ChatCompletionResult(
            request_id="request-id",
            generation_id="generation-id",
            model=model,
            content=content,
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


def make_task(tmp_path: Path) -> TaskSpec:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    return TaskSpec(
        task_id="stable-task",
        instruction="Confirm the repository is valid.",
        source_dir=source,
        test_command=(
            "python",
            "-c",
            "import app; assert app.VALUE == 1",
        ),
        timeout_seconds=30,
    )


def test_benchmark_runs_each_strategy_and_repetition(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "runs.jsonl"
    config = BenchmarkConfig(
        model="test/model",
        strategies=(
            ExecutionStrategy.SINGLE,
            ExecutionStrategy.SEQUENTIAL,
        ),
        repetitions=2,
    )
    runner = BenchmarkRunner(
        FakeChatClient(),
        WorkspaceManager(tmp_path / "workspaces"),
        config,
        writer=JsonlRunWriter(output),
    )

    result = runner.run(
        make_task(tmp_path),
        experiment_id="experiment-42",
    )

    assert result.experiment_id == "experiment-42"
    assert result.task_id == "stable-task"
    assert result.total_runs == 4
    assert result.passed_runs == 4
    assert result.pass_rate == 1.0

    assert [run.strategy for run in result.runs] == [
        ExecutionStrategy.SINGLE,
        ExecutionStrategy.SINGLE,
        ExecutionStrategy.SEQUENTIAL,
        ExecutionStrategy.SEQUENTIAL,
    ]
    assert [run.run_id for run in result.runs] == [
        "experiment-42-stable-task-single-001",
        "experiment-42-stable-task-single-002",
        "experiment-42-stable-task-sequential-001",
        "experiment-42-stable-task-sequential-002",
    ]
    assert len(result.runs_for(ExecutionStrategy.SINGLE)) == 2
    assert len(result.runs_for(ExecutionStrategy.SEQUENTIAL)) == 2

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 4
    assert [record["model_call_count"] for record in records] == [
        1,
        1,
        2,
        2,
    ]
    assert all(record["passed"] for record in records)

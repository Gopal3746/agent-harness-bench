from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_harness import (
    BenchmarkConfig,
    ChatCompletionResult,
    ExecutionStrategy,
    SingleTaskRunner,
    StreamedToolCall,
    TaskSpec,
    WorkspaceManager,
)


class FakeChatClient:
    def __init__(
        self,
        responses: list[ChatCompletionResult],
    ) -> None:
        self.responses = responses
        self.session_ids: list[str | None] = []

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
        del messages, model, tools, temperature, max_tokens, seed
        self.session_ids.append(session_id)
        return self.responses.pop(0)


def make_completion(
    *,
    content: str = "",
    tool_calls: tuple[StreamedToolCall, ...] = (),
) -> ChatCompletionResult:
    return ChatCompletionResult(
        request_id="request-id",
        generation_id="generation-id",
        model="test/model",
        content=content,
        reasoning="",
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
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
        task_id="update-value",
        instruction="Change VALUE from 1 to 2.",
        source_dir=source,
        test_command=(
            "python",
            "-c",
            (
                "from pathlib import Path; "
                "assert Path('app.py').read_text() == "
                "'VALUE = 2\\n'"
            ),
        ),
        timeout_seconds=30,
    )


def test_runner_edits_isolated_workspace_and_verifies_task(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="call-1",
                        name="replace_text",
                        arguments=(
                            '{"path":"app.py",'
                            '"old":"VALUE = 1",'
                            '"new":"VALUE = 2"}'
                        ),
                    ),
                )
            ),
            make_completion(content="The change is complete."),
        ]
    )
    workspaces = WorkspaceManager(
        tmp_path / "workspaces",
        retain_workspaces=True,
    )
    config = BenchmarkConfig(
        model="test/model",
        repetitions=1,
    )

    result = SingleTaskRunner(
        client,
        workspaces,
        config,
    ).run(task, run_id="run-001")

    assert result.passed is True
    assert result.verification.succeeded is True
    assert result.strategy is ExecutionStrategy.SINGLE
    assert result.run_id == "run-001"
    assert client.session_ids == ["run-001", "run-001"]
    assert (result.workspace.path / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert (task.source_dir / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"


def test_runner_fails_when_independent_verification_fails(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    client = FakeChatClient(
        [make_completion(content="No changes are needed.")]
    )
    workspaces = WorkspaceManager(
        tmp_path / "workspaces",
        retain_workspaces=True,
    )
    config = BenchmarkConfig(
        model="test/model",
        repetitions=1,
    )

    result = SingleTaskRunner(
        client,
        workspaces,
        config,
    ).run(task)

    assert result.agent.completed is True
    assert result.verification.succeeded is False
    assert result.passed is False


def test_runner_removes_workspace_by_default(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    client = FakeChatClient(
        [make_completion(content="Finished.")]
    )
    runner = SingleTaskRunner(
        client,
        WorkspaceManager(tmp_path / "workspaces"),
        BenchmarkConfig(
            model="test/model",
            repetitions=1,
        ),
    )

    result = runner.run(task, run_id="temporary-run")

    assert not result.workspace.path.exists()


def test_runner_requires_single_strategy(
    tmp_path: Path,
) -> None:
    config = BenchmarkConfig(
        model="test/model",
        strategies=(ExecutionStrategy.SEQUENTIAL,),
        repetitions=1,
    )

    with pytest.raises(
        ValueError,
        match="must include the single strategy",
    ):
        SingleTaskRunner(
            FakeChatClient([]),
            WorkspaceManager(tmp_path / "workspaces"),
            config,
        )

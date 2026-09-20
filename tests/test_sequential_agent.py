from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_harness import (
    AgentStatus,
    ChatCompletionResult,
    CommandRunner,
    OpenRouterError,
    RepositoryTools,
    SequentialAgent,
    StreamedToolCall,
    TaskSpec,
    TokenUsage,
    ToolDispatcher,
)


class FakeChatClient:
    def __init__(
        self,
        responses: list[
            ChatCompletionResult | OpenRouterError
        ],
    ) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

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
        self.requests.append(
            {
                "messages": list(messages),
                "model": model,
                "tools": tools,
                "session_id": session_id,
            }
        )
        response = self.responses.pop(0)

        if isinstance(response, OpenRouterError):
            raise response

        return response


def make_completion(
    content: str = "",
    *,
    tool_calls: tuple[StreamedToolCall, ...] = (),
    usage: TokenUsage | None = None,
) -> ChatCompletionResult:
    return ChatCompletionResult(
        request_id="request-id",
        generation_id="generation-id",
        model="test/model",
        content=content,
        reasoning="",
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage,
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


def make_dependencies(
    tmp_path: Path,
) -> tuple[TaskSpec, ToolDispatcher, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    task = TaskSpec(
        task_id="update-value",
        instruction="Update and inspect the value.",
        source_dir=repository,
        test_command=("python", "-c", "print('ok')"),
    )

    dispatcher = ToolDispatcher(
        repository=RepositoryTools(repository),
        commands=CommandRunner(repository),
        task=task,
    )

    return task, dispatcher, repository


def plan_completion() -> ChatCompletionResult:
    return make_completion(
        json.dumps(
            {
                "subtasks": [
                    {
                        "subtask_id": "edit",
                        "instruction": "Change VALUE to 2.",
                        "depends_on": [],
                    },
                    {
                        "subtask_id": "inspect",
                        "instruction": "Inspect the updated value.",
                        "depends_on": ["edit"],
                    },
                ]
            }
        ),
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=0,
            reasoning_tokens=0,
            cost_usd=0.01,
        ),
    )


def test_sequential_agent_runs_subtasks_in_shared_workspace(
    tmp_path: Path,
) -> None:
    task, dispatcher, repository = make_dependencies(tmp_path)
    client = FakeChatClient(
        [
            plan_completion(),
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="edit-call",
                        name="replace_text",
                        arguments=json.dumps(
                            {
                                "path": "app.py",
                                "old": "VALUE = 1",
                                "new": "VALUE = 2",
                            }
                        ),
                    ),
                )
            ),
            make_completion("Value updated."),
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="read-call",
                        name="read_file",
                        arguments='{"path":"app.py"}',
                    ),
                )
            ),
            make_completion("Updated value inspected."),
        ]
    )

    agent = SequentialAgent(
        client,
        dispatcher,
        model="test/model",
        session_id="run-001",
    )

    result = agent.run(task)

    assert result.status is AgentStatus.COMPLETED
    assert result.completed is True
    assert [
        item.subtask.subtask_id for item in result.subtasks
    ] == ["edit", "inspect"]

    assert (repository / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"

    assert "VALUE = 2" in client.requests[4]["messages"][-1][
        "content"
    ]

    assert [
        request["session_id"] for request in client.requests
    ] == [
        "run-001-planner",
        "run-001-edit",
        "run-001-edit",
        "run-001-inspect",
        "run-001-inspect",
    ]

    assert "app.py" in client.requests[0]["messages"][1][
        "content"
    ]


def test_sequential_agent_aggregates_calls_and_usage(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_dependencies(tmp_path)

    usage = TokenUsage(
        input_tokens=20,
        output_tokens=4,
        total_tokens=24,
        cached_tokens=0,
        reasoning_tokens=0,
        cost_usd=0.02,
    )

    client = FakeChatClient(
        [
            plan_completion(),
            make_completion("First complete.", usage=usage),
            make_completion("Second complete.", usage=usage),
        ]
    )

    result = SequentialAgent(
        client,
        dispatcher,
        model="test/model",
    ).run(task)

    assert result.total_model_calls == 3
    assert result.total_tool_calls == 0
    assert result.total_input_tokens == 50
    assert result.total_output_tokens == 13
    assert result.total_cost_usd == 0.05
    assert result.total_model_latency_ms == 450


def test_sequential_agent_stops_after_failed_subtask(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_dependencies(tmp_path)

    client = FakeChatClient(
        [
            plan_completion(),
            OpenRouterError("provider unavailable"),
            make_completion("must not be used"),
        ]
    )

    result = SequentialAgent(
        client,
        dispatcher,
        model="test/model",
    ).run(task)

    assert result.status is AgentStatus.FAILED
    assert result.completed is False
    assert len(result.subtasks) == 1
    assert result.subtasks[0].subtask.subtask_id == "edit"
    assert "subtask 'edit' failed" in (result.error or "")
    assert len(client.requests) == 2


def test_sequential_agent_reports_planning_failure(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_dependencies(tmp_path)
    client = FakeChatClient(
        [make_completion("not valid JSON")]
    )

    result = SequentialAgent(
        client,
        dispatcher,
        model="test/model",
    ).run(task)

    assert result.status is AgentStatus.FAILED
    assert result.planning is None
    assert result.subtasks == ()
    assert "invalid task plan" in (result.error or "")

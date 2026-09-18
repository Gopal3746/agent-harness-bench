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
    SingleAgent,
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
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "session_id": session_id,
            }
        )

        response = self.responses.pop(0)

        if isinstance(response, OpenRouterError):
            raise response

        return response


def make_completion(
    *,
    content: str = "",
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
        finish_reason=(
            "tool_calls"
            if tool_calls
            else "stop"
        ),
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


def make_agent_dependencies(
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
        instruction="Change VALUE from 1 to 2.",
        source_dir=repository,
        test_command=(
            "python",
            "-c",
            "print('tests passed')",
        ),
        timeout_seconds=30,
    )

    dispatcher = ToolDispatcher(
        repository=RepositoryTools(repository),
        commands=CommandRunner(repository),
        task=task,
    )

    return task, dispatcher, repository


def test_agent_executes_tool_then_returns_answer(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )

    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="call-1",
                        name="list_files",
                        arguments="{}",
                    ),
                )
            ),
            make_completion(
                content="The repository was inspected."
            ),
        ]
    )

    times = iter(
        [
            1_000_000_000,
            1_500_000_000,
        ]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
        clock_ns=lambda: next(times),
    )

    result = agent.run(task)

    assert result.status is AgentStatus.COMPLETED
    assert result.completed is True
    assert result.final_answer == (
        "The repository was inspected."
    )
    assert result.total_model_calls == 2
    assert result.total_tool_calls == 1
    assert result.wall_time_ms == 500

    second_request = client.requests[1]
    roles = [
        message["role"]
        for message in second_request["messages"]
    ]

    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
    ]

    tool_message = second_request["messages"][-1]
    assert json.loads(tool_message["content"]) == [
        "app.py"
    ]


def test_agent_returns_tool_error_to_model(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )

    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="call-missing",
                        name="read_file",
                        arguments=(
                            '{"path":"missing.py"}'
                        ),
                    ),
                )
            ),
            make_completion(
                content="The requested file does not exist."
            ),
        ]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
    )

    result = agent.run(task)

    assert result.completed is True
    assert result.tool_calls[0].is_error is True

    tool_message = client.requests[1]["messages"][-1]
    assert "file does not exist" in tool_message["content"]


def test_agent_stops_at_maximum_steps(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )

    repeated_call = make_completion(
        tool_calls=(
            StreamedToolCall(
                index=0,
                call_id="",
                name="list_files",
                arguments="{}",
            ),
        )
    )

    client = FakeChatClient(
        [
            repeated_call,
            repeated_call,
        ]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
        max_steps=2,
    )

    result = agent.run(task)

    assert result.status is AgentStatus.MAX_STEPS
    assert result.completed is False
    assert result.steps == 2
    assert result.total_model_calls == 2
    assert result.total_tool_calls == 2
    assert "maximum of 2" in (result.error or "")


def test_agent_aggregates_usage_metrics(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )

    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="call-1",
                        name="list_files",
                        arguments="{}",
                    ),
                ),
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=3,
                    total_tokens=13,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    cost_usd=0.01,
                ),
            ),
            make_completion(
                content="Finished.",
                usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=4,
                    total_tokens=24,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    cost_usd=0.02,
                ),
            ),
        ]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
    )

    result = agent.run(task)

    assert result.total_input_tokens == 30
    assert result.total_output_tokens == 7
    assert result.total_cost_usd == 0.03
    assert result.total_model_latency_ms == 300


def test_agent_passes_configuration_and_prompt(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )
    client = FakeChatClient(
        [make_completion(content="Finished.")]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
        temperature=0.2,
        max_tokens=500,
        seed=7,
        session_id="run-123",
    )

    agent.run(task)

    request = client.requests[0]

    assert request["model"] == "test/model"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 500
    assert request["seed"] == 7
    assert request["session_id"] == "run-123"
    assert len(request["tools"]) == 7

    messages = request["messages"]
    assert messages[0]["role"] == "system"
    assert "isolated repository" in messages[0]["content"]
    assert task.instruction in messages[1]["content"]
    assert "verification command" in messages[1]["content"]


def test_agent_records_model_failure(
    tmp_path: Path,
) -> None:
    task, dispatcher, _ = make_agent_dependencies(
        tmp_path
    )

    client = FakeChatClient(
        [OpenRouterError("provider unavailable")]
    )

    agent = SingleAgent(
        client,
        dispatcher,
        model="test/model",
    )

    result = agent.run(task)

    assert result.status is AgentStatus.FAILED
    assert result.completed is False
    assert result.error == "provider unavailable"
    assert result.total_model_calls == 0
    assert result.total_tool_calls == 0

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_harness import (
    AgentStatus,
    ChatCompletionResult,
    CommandRunner,
    OpenRouterError,
    ParallelAgent,
    RepositoryTools,
    StreamedToolCall,
    TaskSpec,
    ToolDispatcher,
)


class RoutingChatClient:
    def __init__(
        self,
        responses: dict[
            str,
            list[ChatCompletionResult | OpenRouterError],
        ],
    ) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

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
        assert session_id is not None

        with self.lock:
            self.requests.append(
                {
                    "messages": list(messages),
                    "model": model,
                    "tools": tools,
                    "session_id": session_id,
                }
            )
            response = self.responses[session_id].pop(0)

        if isinstance(response, OpenRouterError):
            raise response

        return response


class ConcurrentChatClient(RoutingChatClient):
    def __init__(
        self,
        responses: dict[
            str,
            list[ChatCompletionResult | OpenRouterError],
        ],
        concurrent_sessions: set[str],
    ) -> None:
        super().__init__(responses)
        self.concurrent_sessions = concurrent_sessions
        self.arrived: set[str] = set()
        self.ready = threading.Event()

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
        assert session_id is not None

        if session_id in self.concurrent_sessions:
            with self.lock:
                self.arrived.add(session_id)

                if self.arrived == self.concurrent_sessions:
                    self.ready.set()

            if not self.ready.wait(timeout=2):
                raise AssertionError(
                    "independent subtasks did not run concurrently"
                )

        return super().complete(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            session_id=session_id,
        )


def make_completion(
    content: str = "",
    *,
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


def make_plan(
    subtasks: list[dict[str, Any]],
) -> ChatCompletionResult:
    return make_completion(
        json.dumps({"subtasks": subtasks})
    )


def tool_completion(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> ChatCompletionResult:
    return make_completion(
        tool_calls=(
            StreamedToolCall(
                index=0,
                call_id=call_id,
                name=name,
                arguments=json.dumps(arguments),
            ),
        )
    )


def make_dependencies(
    tmp_path: Path,
) -> tuple[TaskSpec, ToolDispatcher, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "users.py").write_text(
        "USER_LIMIT = 1\n",
        encoding="utf-8",
    )
    (repository / "network.py").write_text(
        "PORT_LIMIT = 1\n",
        encoding="utf-8",
    )

    task = TaskSpec(
        task_id="independent-updates",
        instruction="Update independent limits.",
        source_dir=repository,
        test_command=("python", "-c", "print('ok')"),
    )

    dispatcher = ToolDispatcher(
        repository=RepositoryTools(repository),
        commands=CommandRunner(repository),
        task=task,
    )

    return task, dispatcher, repository


def independent_plan() -> ChatCompletionResult:
    return make_plan(
        [
            {
                "subtask_id": "users",
                "instruction": "Update the user limit.",
                "depends_on": [],
            },
            {
                "subtask_id": "network",
                "instruction": "Update the network limit.",
                "depends_on": [],
            },
            {
                "subtask_id": "verify",
                "instruction": "Inspect the combined changes.",
                "depends_on": ["users", "network"],
            },
        ]
    )


def test_parallel_agent_runs_and_merges_dependency_groups(
    tmp_path: Path,
) -> None:
    task, dispatcher, repository = make_dependencies(tmp_path)

    client = ConcurrentChatClient(
        {
            "run-planner": [independent_plan()],
            "run-users": [
                tool_completion(
                    "users-edit",
                    "replace_text",
                    {
                        "path": "users.py",
                        "old": "USER_LIMIT = 1",
                        "new": "USER_LIMIT = 2",
                    },
                ),
                make_completion("Users complete."),
            ],
            "run-network": [
                tool_completion(
                    "network-edit",
                    "replace_text",
                    {
                        "path": "network.py",
                        "old": "PORT_LIMIT = 1",
                        "new": "PORT_LIMIT = 2",
                    },
                ),
                make_completion("Network complete."),
            ],
            "run-verify": [
                tool_completion(
                    "verify-users",
                    "read_file",
                    {"path": "users.py"},
                ),
                make_completion("Combined changes verified."),
            ],
        },
        concurrent_sessions={"run-users", "run-network"},
    )

    result = ParallelAgent(
        client,
        dispatcher,
        model="test/model",
        parallelism=2,
        session_id="run",
    ).run(task)

    assert result.status is AgentStatus.COMPLETED
    assert result.completed is True

    assert [
        [item.subtask.subtask_id for item in group]
        for group in result.groups
    ] == [["users", "network"], ["verify"]]

    assert client.arrived == {
        "run-users",
        "run-network",
    }

    assert (repository / "users.py").read_text(
        encoding="utf-8"
    ) == "USER_LIMIT = 2\n"

    assert (repository / "network.py").read_text(
        encoding="utf-8"
    ) == "PORT_LIMIT = 2\n"

    verify_requests = [
        request
        for request in client.requests
        if request["session_id"] == "run-verify"
    ]

    assert "USER_LIMIT = 2" in verify_requests[-1][
        "messages"
    ][-1]["content"]

    assert result.total_model_calls == 7
    assert result.total_tool_calls == 3


def test_parallel_agent_rejects_conflicting_changes(
    tmp_path: Path,
) -> None:
    task, dispatcher, repository = make_dependencies(tmp_path)

    plan = make_plan(
        [
            {
                "subtask_id": "first",
                "instruction": "Set the user limit to 2.",
                "depends_on": [],
            },
            {
                "subtask_id": "second",
                "instruction": "Set the user limit to 3.",
                "depends_on": [],
            },
        ]
    )

    client = RoutingChatClient(
        {
            "run-planner": [plan],
            "run-first": [
                tool_completion(
                    "first-edit",
                    "replace_text",
                    {
                        "path": "users.py",
                        "old": "USER_LIMIT = 1",
                        "new": "USER_LIMIT = 2",
                    },
                ),
                make_completion("First complete."),
            ],
            "run-second": [
                tool_completion(
                    "second-edit",
                    "replace_text",
                    {
                        "path": "users.py",
                        "old": "USER_LIMIT = 1",
                        "new": "USER_LIMIT = 3",
                    },
                ),
                make_completion("Second complete."),
            ],
        }
    )

    result = ParallelAgent(
        client,
        dispatcher,
        model="test/model",
        parallelism=2,
        session_id="run",
    ).run(task)

    assert result.status is AgentStatus.FAILED
    assert "conflicting changes" in (result.error or "")
    assert "users.py" in (result.error or "")

    assert (repository / "users.py").read_text(
        encoding="utf-8"
    ) == "USER_LIMIT = 1\n"


def test_parallel_agent_does_not_merge_failed_group(
    tmp_path: Path,
) -> None:
    task, dispatcher, repository = make_dependencies(tmp_path)

    client = RoutingChatClient(
        {
            "run-planner": [independent_plan()],
            "run-users": [
                tool_completion(
                    "users-edit",
                    "replace_text",
                    {
                        "path": "users.py",
                        "old": "USER_LIMIT = 1",
                        "new": "USER_LIMIT = 2",
                    },
                ),
                make_completion("Users complete."),
            ],
            "run-network": [
                OpenRouterError("provider unavailable")
            ],
            "run-verify": [
                make_completion("must not run")
            ],
        }
    )

    result = ParallelAgent(
        client,
        dispatcher,
        model="test/model",
        parallelism=2,
        session_id="run",
    ).run(task)

    assert result.status is AgentStatus.FAILED
    assert "subtask 'network' failed" in (result.error or "")
    assert len(result.groups) == 1

    assert (repository / "users.py").read_text(
        encoding="utf-8"
    ) == "USER_LIMIT = 1\n"

    assert not any(
        request["session_id"] == "run-verify"
        for request in client.requests
    )


def test_parallel_agent_requires_multiple_workers(
    tmp_path: Path,
) -> None:
    _, dispatcher, _ = make_dependencies(tmp_path)
    client = RoutingChatClient({})

    with pytest.raises(ValueError, match="at least 2"):
        ParallelAgent(
            client,
            dispatcher,
            model="test/model",
            parallelism=1,
        )

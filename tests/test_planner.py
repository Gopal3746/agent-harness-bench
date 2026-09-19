from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_harness import (
    ChatCompletionResult,
    OpenRouterError,
    PlanningError,
    SubtaskSpec,
    TaskPlan,
    TaskPlanner,
    TaskSpec,
)


class FakeChatClient:
    def __init__(
        self,
        response: ChatCompletionResult | OpenRouterError,
    ) -> None:
        self.response = response
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

        if isinstance(self.response, OpenRouterError):
            raise self.response

        return self.response


def make_completion(content: str) -> ChatCompletionResult:
    return ChatCompletionResult(
        request_id="request-id",
        generation_id="generation-id",
        model="test/model",
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
    return TaskSpec(
        task_id="validation-task",
        instruction="Add validation across independent modules.",
        source_dir=tmp_path,
    )


def test_plan_exposes_sequential_order_and_parallel_groups() -> None:
    plan = TaskPlan(
        subtasks=(
            SubtaskSpec(
                subtask_id="users",
                instruction="Update user validation.",
            ),
            SubtaskSpec(
                subtask_id="network",
                instruction="Update port validation.",
            ),
            SubtaskSpec(
                subtask_id="integration",
                instruction="Check the combined behavior.",
                depends_on=("users", "network"),
            ),
        )
    )

    assert [
        subtask.subtask_id for subtask in plan.execution_order()
    ] == ["users", "network", "integration"]
    assert [
        [subtask.subtask_id for subtask in group]
        for group in plan.parallel_groups()
    ] == [["users", "network"], ["integration"]]


def test_plan_rejects_duplicate_subtask_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        TaskPlan(
            subtasks=(
                SubtaskSpec(
                    subtask_id="users",
                    instruction="First change.",
                ),
                SubtaskSpec(
                    subtask_id="users",
                    instruction="Second change.",
                ),
            )
        )


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown dependencies"):
        TaskPlan(
            subtasks=(
                SubtaskSpec(
                    subtask_id="users",
                    instruction="Update user validation.",
                    depends_on=("missing",),
                ),
            )
        )


def test_plan_rejects_dependency_cycle() -> None:
    with pytest.raises(ValidationError, match="contains a cycle"):
        TaskPlan(
            subtasks=(
                SubtaskSpec(
                    subtask_id="users",
                    instruction="Update user validation.",
                    depends_on=("network",),
                ),
                SubtaskSpec(
                    subtask_id="network",
                    instruction="Update port validation.",
                    depends_on=("users",),
                ),
            )
        )


def test_planner_parses_json_and_records_request(
    tmp_path: Path,
) -> None:
    response = make_completion(
        "```json\n"
        + json.dumps(
            {
                "subtasks": [
                    {
                        "subtask_id": "users",
                        "instruction": "Update user validation.",
                        "depends_on": [],
                    },
                    {
                        "subtask_id": "network",
                        "instruction": "Update port validation.",
                        "depends_on": [],
                    },
                ]
            }
        )
        + "\n```"
    )
    client = FakeChatClient(response)
    times = iter([1_000_000_000, 1_250_000_000])
    planner = TaskPlanner(
        client,
        model="test/model",
        max_subtasks=4,
        session_id="run-001-planner",
        clock_ns=lambda: next(times),
    )

    result = planner.plan(
        make_task(tmp_path),
        repository_files=(
            "validation_app/users.py",
            "validation_app/network.py",
        ),
    )

    assert len(result.plan.subtasks) == 2
    assert result.wall_time_ms == 250.0

    request = client.requests[0]
    assert request["model"] == "test/model"
    assert request["tools"] is None
    assert request["session_id"] == "run-001-planner"
    assert "validation_app/users.py" in request["messages"][1]["content"]
    assert "at most 4 subtasks" in request["messages"][1]["content"]


def test_planner_rejects_invalid_model_output(
    tmp_path: Path,
) -> None:
    planner = TaskPlanner(
        FakeChatClient(make_completion("not JSON")),
        model="test/model",
    )

    with pytest.raises(PlanningError, match="invalid task plan"):
        planner.plan(make_task(tmp_path))


def test_planner_wraps_model_errors(tmp_path: Path) -> None:
    planner = TaskPlanner(
        FakeChatClient(OpenRouterError("provider unavailable")),
        model="test/model",
    )

    with pytest.raises(PlanningError, match="provider unavailable"):
        planner.plan(make_task(tmp_path))

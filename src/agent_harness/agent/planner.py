from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agent_harness.agent.single import ChatClient
from agent_harness.llm import ChatCompletionResult, OpenRouterError
from agent_harness.models import TaskSpec

_MAX_PLAN_SUBTASKS = 8

_PLANNER_SYSTEM_PROMPT = """\
You decompose repository-level coding tasks into small implementation subtasks.

Return only a JSON object with a `subtasks` array. Each subtask must have:
- `subtask_id`: a short lowercase identifier
- `instruction`: a precise implementation instruction
- `depends_on`: an array of prerequisite subtask IDs

Create independent subtasks when work can safely happen in separate files or
modules. Add dependencies only when one change truly requires another. Do not
include a final testing subtask because the harness performs independent
verification after execution. Do not return Markdown or explanatory text.
"""


class PlanningError(RuntimeError):
    """A model response could not produce a valid task plan."""


class SubtaskSpec(BaseModel):
    """One independently executable unit of a coding task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subtask_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    instruction: str
    depends_on: tuple[str, ...] = ()

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("instruction must not be blank")

        return cleaned

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        cleaned = tuple(dependency.strip() for dependency in value)

        if any(not dependency for dependency in cleaned):
            raise ValueError("dependencies must not be blank")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("dependencies must be unique")

        return cleaned


class TaskPlan(BaseModel):
    """Validated dependency graph produced by the planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subtasks: tuple[SubtaskSpec, ...] = Field(
        min_length=1,
        max_length=_MAX_PLAN_SUBTASKS,
    )

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        identifiers = [subtask.subtask_id for subtask in self.subtasks]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("subtask IDs must be unique")

        known_identifiers = set(identifiers)

        for subtask in self.subtasks:
            if subtask.subtask_id in subtask.depends_on:
                raise ValueError(
                    f"subtask {subtask.subtask_id!r} cannot depend on itself"
                )

            unknown = set(subtask.depends_on) - known_identifiers

            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"subtask {subtask.subtask_id!r} has unknown "
                    f"dependencies: {names}"
                )

        self.parallel_groups()
        return self

    def parallel_groups(self) -> tuple[tuple[SubtaskSpec, ...], ...]:
        """Return stable topological layers that can run concurrently."""

        completed: set[str] = set()
        remaining = list(self.subtasks)
        groups: list[tuple[SubtaskSpec, ...]] = []

        while remaining:
            ready = tuple(
                subtask
                for subtask in remaining
                if set(subtask.depends_on) <= completed
            )

            if not ready:
                raise ValueError("subtask dependency graph contains a cycle")

            groups.append(ready)
            ready_ids = {subtask.subtask_id for subtask in ready}
            completed.update(ready_ids)
            remaining = [
                subtask
                for subtask in remaining
                if subtask.subtask_id not in ready_ids
            ]

        return tuple(groups)

    def execution_order(self) -> tuple[SubtaskSpec, ...]:
        """Return a stable dependency-respecting sequential order."""

        return tuple(
            subtask
            for group in self.parallel_groups()
            for subtask in group
        )


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Validated plan and inference trace from one planner call."""

    plan: TaskPlan
    model_call: ChatCompletionResult
    wall_time_ms: float


class TaskPlanner:
    """Uses a model to decompose one coding task."""

    def __init__(
        self,
        client: ChatClient,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 2_000,
        seed: int | None = 42,
        max_subtasks: int = _MAX_PLAN_SUBTASKS,
        session_id: str | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")

        if not 1 <= max_subtasks <= _MAX_PLAN_SUBTASKS:
            raise ValueError(
                f"max_subtasks must be between 1 and {_MAX_PLAN_SUBTASKS}"
            )

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_subtasks = max_subtasks
        self.session_id = session_id
        self.clock_ns = clock_ns

    def plan(
        self,
        task: TaskSpec,
        *,
        repository_files: Sequence[str] = (),
    ) -> PlanningResult:
        """Request and validate a dependency-aware task plan."""

        started_ns = self.clock_ns()

        try:
            completion = self.client.complete(
                self._messages(task, repository_files),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                session_id=self.session_id,
            )
        except OpenRouterError as error:
            raise PlanningError(
                f"planner model request failed: {error}"
            ) from error

        plan = self._parse_plan(completion.content)

        if len(plan.subtasks) > self.max_subtasks:
            raise PlanningError(
                f"planner returned {len(plan.subtasks)} subtasks; "
                f"maximum is {self.max_subtasks}"
            )

        return PlanningResult(
            plan=plan,
            model_call=completion,
            wall_time_ms=(self.clock_ns() - started_ns) / 1_000_000,
        )

    def _messages(
        self,
        task: TaskSpec,
        repository_files: Sequence[str],
    ) -> list[dict[str, Any]]:
        files = (
            "\n".join(f"- {path}" for path in repository_files)
            or "- (not provided)"
        )

        return [
            {
                "role": "system",
                "content": _PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Task ID: {task.task_id}\n\n"
                    f"Instruction:\n{task.instruction}\n\n"
                    f"Repository files:\n{files}\n\n"
                    f"Create at most {self.max_subtasks} subtasks."
                ),
            },
        ]

    @staticmethod
    def _parse_plan(content: str) -> TaskPlan:
        candidate = content.strip()

        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()

            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(candidate)
            return TaskPlan.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as error:
            raise PlanningError(
                f"planner returned an invalid task plan: {error}"
            ) from error

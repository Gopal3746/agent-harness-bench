from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_harness.agent.planner import (
    PlanningError,
    PlanningResult,
    SubtaskSpec,
    TaskPlanner,
)
from agent_harness.agent.single import (
    AgentRunResult,
    AgentStatus,
    ChatClient,
    SingleAgent,
    ToolCallRecord,
)
from agent_harness.llm import ChatCompletionResult
from agent_harness.models import TaskSpec
from agent_harness.tools import ToolDispatcher


@dataclass(frozen=True, slots=True)
class SubtaskRunResult:
    """One planned subtask and its single-agent execution trace."""

    subtask: SubtaskSpec
    agent: AgentRunResult


@dataclass(frozen=True, slots=True)
class SequentialRunResult:
    """Planning and execution trace for a sequential-subagent run."""

    task_id: str
    status: AgentStatus
    planning: PlanningResult | None
    subtasks: tuple[SubtaskRunResult, ...]
    error: str | None
    wall_time_ms: float

    @property
    def completed(self) -> bool:
        return self.status is AgentStatus.COMPLETED

    @property
    def model_calls(self) -> tuple[ChatCompletionResult, ...]:
        planning_calls = (
            ()
            if self.planning is None
            else (self.planning.model_call,)
        )

        return planning_calls + tuple(
            model_call
            for subtask in self.subtasks
            for model_call in subtask.agent.model_calls
        )

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        return tuple(
            tool_call
            for subtask in self.subtasks
            for tool_call in subtask.agent.tool_calls
        )

    @property
    def total_model_calls(self) -> int:
        return len(self.model_calls)

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(
            call.usage.input_tokens
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            call.usage.output_tokens
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(
            call.usage.cost_usd or 0.0
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_model_latency_ms(self) -> float:
        return sum(
            call.total_latency_ms
            for call in self.model_calls
        )

    @property
    def total_tool_duration_ms(self) -> float:
        return sum(
            call.duration_ms
            for call in self.tool_calls
        )


class SequentialAgent:
    """Plans a task and executes its subtasks one at a time."""

    def __init__(
        self,
        client: ChatClient,
        dispatcher: ToolDispatcher,
        *,
        model: str,
        max_steps: int = 20,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = 42,
        max_subtasks: int = 8,
        session_id: str | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.client = client
        self.dispatcher = dispatcher
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_subtasks = max_subtasks
        self.session_id = session_id
        self.clock_ns = clock_ns

    def run(self, task: TaskSpec) -> SequentialRunResult:
        """Plan and execute subtasks in dependency-respecting order."""

        started_ns = self.clock_ns()
        planner = TaskPlanner(
            self.client,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            max_subtasks=self.max_subtasks,
            session_id=self._session_id("planner"),
            clock_ns=self.clock_ns,
        )

        try:
            planning = planner.plan(
                task,
                repository_files=self.dispatcher.repository.list_files(),
            )
        except PlanningError as error:
            return self._result(
                task=task,
                status=AgentStatus.FAILED,
                planning=None,
                subtasks=(),
                error=str(error),
                started_ns=started_ns,
            )

        completed: list[SubtaskRunResult] = []

        for subtask in planning.plan.execution_order():
            agent = SingleAgent(
                self.client,
                self.dispatcher,
                model=self.model,
                max_steps=self.max_steps,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                session_id=self._session_id(subtask.subtask_id),
                clock_ns=self.clock_ns,
            )
            agent_result = agent.run(
                self._subtask_task(task, subtask)
            )
            completed.append(
                SubtaskRunResult(
                    subtask=subtask,
                    agent=agent_result,
                )
            )

            if not agent_result.completed:
                return self._result(
                    task=task,
                    status=agent_result.status,
                    planning=planning,
                    subtasks=tuple(completed),
                    error=(
                        f"subtask {subtask.subtask_id!r} failed: "
                        f"{agent_result.error or 'unknown error'}"
                    ),
                    started_ns=started_ns,
                )

        return self._result(
            task=task,
            status=AgentStatus.COMPLETED,
            planning=planning,
            subtasks=tuple(completed),
            error=None,
            started_ns=started_ns,
        )

    def _session_id(self, suffix: str) -> str | None:
        if self.session_id is None:
            return None

        return f"{self.session_id}-{suffix}"

    @staticmethod
    def _subtask_task(
        task: TaskSpec,
        subtask: SubtaskSpec,
    ) -> TaskSpec:
        dependencies = (
            ", ".join(subtask.depends_on)
            if subtask.depends_on
            else "none"
        )

        return task.model_copy(
            update={
                "instruction": (
                    "Complete this subtask of the larger task:\n"
                    f"{subtask.instruction}\n\n"
                    f"Subtask ID: {subtask.subtask_id}\n"
                    f"Completed prerequisites: {dependencies}\n\n"
                    "Work only on this subtask. Preserve changes already "
                    "made in the shared workspace by earlier subtasks."
                )
            }
        )

    def _result(
        self,
        *,
        task: TaskSpec,
        status: AgentStatus,
        planning: PlanningResult | None,
        subtasks: tuple[SubtaskRunResult, ...],
        error: str | None,
        started_ns: int,
    ) -> SequentialRunResult:
        return SequentialRunResult(
            task_id=task.task_id,
            status=status,
            planning=planning,
            subtasks=subtasks,
            error=error,
            wall_time_ms=(
                self.clock_ns() - started_ns
            )
            / 1_000_000,
        )

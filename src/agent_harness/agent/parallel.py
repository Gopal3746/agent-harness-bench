from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from agent_harness.agent.merge import (
    WorkspaceChangeSet,
    WorkspaceMergeError,
    WorkspaceSnapshot,
    clone_workspace,
    merge_change_sets,
)
from agent_harness.agent.planner import (
    PlanningError,
    PlanningResult,
    SubtaskSpec,
    TaskPlanner,
)
from agent_harness.agent.sequential import SubtaskRunResult
from agent_harness.agent.single import (
    AgentStatus,
    ChatClient,
    SingleAgent,
    ToolCallRecord,
)
from agent_harness.llm import ChatCompletionResult
from agent_harness.models import TaskSpec
from agent_harness.tools import (
    CommandRunner,
    RepositoryTools,
    ToolDispatcher,
)


@dataclass(frozen=True, slots=True)
class _SubtaskOutcome:
    result: SubtaskRunResult
    changes: WorkspaceChangeSet


@dataclass(frozen=True, slots=True)
class ParallelRunResult:
    """Planning and execution trace for a parallel-subagent run."""

    task_id: str
    status: AgentStatus
    planning: PlanningResult | None
    groups: tuple[tuple[SubtaskRunResult, ...], ...]
    error: str | None
    wall_time_ms: float

    @property
    def completed(self) -> bool:
        return self.status is AgentStatus.COMPLETED

    @property
    def subtasks(self) -> tuple[SubtaskRunResult, ...]:
        return tuple(
            subtask
            for group in self.groups
            for subtask in group
        )

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


class ParallelAgent:
    """Executes each dependency-ready subtask group concurrently."""

    def __init__(
        self,
        client: ChatClient,
        dispatcher: ToolDispatcher,
        *,
        model: str,
        parallelism: int = 2,
        max_steps: int = 20,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = 42,
        max_subtasks: int = 8,
        session_id: str | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if parallelism < 2:
            raise ValueError("parallelism must be at least 2")

        self.client = client
        self.dispatcher = dispatcher
        self.model = model
        self.parallelism = parallelism
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_subtasks = max_subtasks
        self.session_id = session_id
        self.clock_ns = clock_ns

    def run(self, task: TaskSpec) -> ParallelRunResult:
        """Plan, execute, and merge dependency-ready subtask groups."""

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
                groups=(),
                error=str(error),
                started_ns=started_ns,
            )

        completed_groups: list[tuple[SubtaskRunResult, ...]] = []

        for group in planning.plan.parallel_groups():
            outcomes = self._run_group(task, group)
            group_results = tuple(
                outcome.result for outcome in outcomes
            )
            completed_groups.append(group_results)

            failed = next(
                (
                    result
                    for result in group_results
                    if not result.agent.completed
                ),
                None,
            )

            if failed is not None:
                return self._result(
                    task=task,
                    status=failed.agent.status,
                    planning=planning,
                    groups=tuple(completed_groups),
                    error=(
                        f"subtask {failed.subtask.subtask_id!r} failed: "
                        f"{failed.agent.error or 'unknown error'}"
                    ),
                    started_ns=started_ns,
                )

            try:
                merge_change_sets(
                    self.dispatcher.repository.root,
                    tuple(outcome.changes for outcome in outcomes),
                )
            except WorkspaceMergeError as error:
                return self._result(
                    task=task,
                    status=AgentStatus.FAILED,
                    planning=planning,
                    groups=tuple(completed_groups),
                    error=str(error),
                    started_ns=started_ns,
                )

        return self._result(
            task=task,
            status=AgentStatus.COMPLETED,
            planning=planning,
            groups=tuple(completed_groups),
            error=None,
            started_ns=started_ns,
        )

    def _run_group(
        self,
        task: TaskSpec,
        group: tuple[SubtaskSpec, ...],
    ) -> tuple[_SubtaskOutcome, ...]:
        baseline = WorkspaceSnapshot.capture(
            self.dispatcher.repository.root
        )

        with tempfile.TemporaryDirectory(
            prefix="agent-harness-parallel-"
        ) as temporary_directory:
            group_root = Path(temporary_directory)

            clones = {
                subtask.subtask_id: clone_workspace(
                    self.dispatcher.repository.root,
                    group_root / subtask.subtask_id,
                )
                for subtask in group
            }

            with ThreadPoolExecutor(
                max_workers=min(self.parallelism, len(group))
            ) as executor:
                futures = [
                    executor.submit(
                        self._run_subtask,
                        task,
                        subtask,
                        clones[subtask.subtask_id],
                        baseline,
                    )
                    for subtask in group
                ]

                return tuple(
                    future.result() for future in futures
                )

    def _run_subtask(
        self,
        task: TaskSpec,
        subtask: SubtaskSpec,
        clone: Path,
        baseline: WorkspaceSnapshot,
    ) -> _SubtaskOutcome:
        cloned_task = task.model_copy(
            update={
                "source_dir": clone,
                "instruction": self._subtask_instruction(subtask),
            }
        )

        dispatcher = self._cloned_dispatcher(
            clone,
            cloned_task,
        )

        agent = SingleAgent(
            self.client,
            dispatcher,
            model=self.model,
            max_steps=self.max_steps,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            session_id=self._session_id(subtask.subtask_id),
            clock_ns=self.clock_ns,
        )

        agent_result = agent.run(cloned_task)
        updated = WorkspaceSnapshot.capture(clone)

        return _SubtaskOutcome(
            result=SubtaskRunResult(
                subtask=subtask,
                agent=agent_result,
            ),
            changes=WorkspaceChangeSet.between(
                subtask.subtask_id,
                baseline,
                updated,
            ),
        )

    def _cloned_dispatcher(
        self,
        clone: Path,
        task: TaskSpec,
    ) -> ToolDispatcher:
        repository = self.dispatcher.repository
        commands = self.dispatcher.commands

        return ToolDispatcher(
            repository=RepositoryTools(
                clone,
                max_file_bytes=repository.max_file_bytes,
                max_read_lines=repository.max_read_lines,
                max_write_bytes=repository.max_write_bytes,
            ),
            commands=CommandRunner(
                clone,
                allowed_commands=commands.allowed_commands,
                default_timeout_seconds=(
                    commands.default_timeout_seconds
                ),
                max_timeout_seconds=commands.max_timeout_seconds,
                max_output_chars=commands.max_output_chars,
            ),
            task=task,
        )

    def _session_id(self, suffix: str) -> str | None:
        if self.session_id is None:
            return None

        return f"{self.session_id}-{suffix}"

    @staticmethod
    def _subtask_instruction(subtask: SubtaskSpec) -> str:
        dependencies = (
            ", ".join(subtask.depends_on)
            if subtask.depends_on
            else "none"
        )

        return (
            "Complete this isolated subtask of the larger task:\n"
            f"{subtask.instruction}\n\n"
            f"Subtask ID: {subtask.subtask_id}\n"
            f"Completed prerequisites: {dependencies}\n\n"
            "Work only on this subtask. Your changes will be merged with "
            "changes from other independent subagents."
        )

    def _result(
        self,
        *,
        task: TaskSpec,
        status: AgentStatus,
        planning: PlanningResult | None,
        groups: tuple[tuple[SubtaskRunResult, ...], ...],
        error: str | None,
        started_ns: int,
    ) -> ParallelRunResult:
        return ParallelRunResult(
            task_id=task.task_id,
            status=status,
            planning=planning,
            groups=groups,
            error=error,
            wall_time_ms=(
                self.clock_ns() - started_ns
            )
            / 1_000_000,
        )

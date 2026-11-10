from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_harness.agent.planner import (
    SubtaskSpec,
    TaskPlan,
)
from agent_harness.models import TaskSpec
from agent_harness.tools import RepositoryTools, ToolExecutionError

_MAX_CONTEXT_CHARS = 12_000
_MAX_FILE_CONTEXT_CHARS = 4_000
_TRUNCATION_MARKER = "\n... file context truncated ..."


def build_worker_task(
    parent_task: TaskSpec,
    subtask: SubtaskSpec,
    plan: TaskPlan,
    *,
    isolated_workspace: bool,
    file_context: Mapping[str, str] | None = None,
) -> TaskSpec:
    """Create a subtask prompt without losing the parent task context."""

    prerequisites = (
        ", ".join(subtask.depends_on)
        if subtask.depends_on
        else "none"
    )
    planned_work = "\n".join(
        _format_plan_item(item, assigned=item == subtask)
        for item in plan.subtasks
    )

    workspace_guidance = (
        "You are working in an isolated clone. Other independent "
        "workers' changes are not present yet and will be merged later."
        if isolated_workspace
        else (
            "You are working in the shared workspace. Changes from "
            "completed prerequisite subtasks may already be present."
        )
    )
    relevant_context = _format_file_context(
        subtask,
        file_context or {},
    )

    instruction = (
        "Original task:\n"
        f"{parent_task.instruction}\n\n"
        "Assigned subtask:\n"
        f"{subtask.instruction}\n\n"
        f"Subtask ID: {subtask.subtask_id}\n"
        f"Available prerequisites: {prerequisites}\n\n"
        "Complete plan and ownership:\n"
        f"{planned_work}\n\n"
        f"{relevant_context}"
        f"{workspace_guidance}\n\n"
        "Work only on the assigned subtask. Do not implement work owned "
        "by another subtask unless it is strictly necessary for your "
        "change. Verify only your subtask with focused commands or "
        "targeted tests. The harness will run the full configured test "
        "suite after all subtask changes are combined."
    )

    return parent_task.model_copy(
        update={"instruction": instruction}
    )


def load_file_context(
    repository: RepositoryTools,
    subtasks: Sequence[SubtaskSpec],
    *,
    max_total_chars: int = _MAX_CONTEXT_CHARS,
    max_file_chars: int = _MAX_FILE_CONTEXT_CHARS,
) -> dict[str, str]:
    """Read each planner-selected file once into a bounded snapshot."""

    if max_total_chars < 1:
        raise ValueError("max_total_chars must be positive")

    if max_file_chars < 1:
        raise ValueError("max_file_chars must be positive")

    paths = tuple(
        dict.fromkeys(
            path
            for subtask in subtasks
            for path in subtask.relevant_files
        )
    )
    context: dict[str, str] = {}
    remaining = max_total_chars
    per_file_limit = (
        min(max_file_chars, max_total_chars // len(paths))
        if paths
        else max_file_chars
    )

    for path in paths:
        if remaining <= 0:
            break

        try:
            content = repository.read_file(path)
        except (OSError, ToolExecutionError) as error:
            content = f"[file unavailable: {error}]"

        limit = min(per_file_limit, remaining)
        context[path] = _truncate(content, limit)
        remaining -= len(context[path])

    return context


def _format_plan_item(
    subtask: SubtaskSpec,
    *,
    assigned: bool,
) -> str:
    owner = " (assigned to you)" if assigned else " (another worker)"
    return (
        f"- {subtask.subtask_id}{owner}: "
        f"{subtask.instruction}"
    )


def _format_file_context(
    subtask: SubtaskSpec,
    file_context: Mapping[str, str],
) -> str:
    sections = [
        f"--- {path} ---\n{file_context[path]}"
        for path in subtask.relevant_files
        if path in file_context
    ]

    if not sections:
        return ""

    return (
        "Relevant repository context (preloaded by the harness):\n"
        + "\n\n".join(sections)
        + "\n\n"
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value

    if limit <= len(_TRUNCATION_MARKER):
        return value[:limit]

    return value[: limit - len(_TRUNCATION_MARKER)] + (
        _TRUNCATION_MARKER
    )

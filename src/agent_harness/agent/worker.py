from __future__ import annotations

from agent_harness.agent.planner import (
    SubtaskSpec,
    TaskPlan,
)
from agent_harness.models import TaskSpec


def build_worker_task(
    parent_task: TaskSpec,
    subtask: SubtaskSpec,
    plan: TaskPlan,
    *,
    isolated_workspace: bool,
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

    instruction = (
        "Original task:\n"
        f"{parent_task.instruction}\n\n"
        "Assigned subtask:\n"
        f"{subtask.instruction}\n\n"
        f"Subtask ID: {subtask.subtask_id}\n"
        f"Available prerequisites: {prerequisites}\n\n"
        "Complete plan and ownership:\n"
        f"{planned_work}\n\n"
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

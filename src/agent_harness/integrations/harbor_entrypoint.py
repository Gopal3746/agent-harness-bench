from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_harness.agent import (
    AgentStatus,
    ChatClient,
    SingleAgent,
)
from agent_harness.llm import OpenRouterClient
from agent_harness.models import TaskSpec
from agent_harness.tools import (
    CommandRunner,
    RepositoryTools,
    ToolDispatcher,
)


class HarborRunSummary(BaseModel):
    """Compact result returned from a Harbor environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentStatus
    final_answer: str
    error: str | None
    steps: int
    wall_time_ms: float
    model_call_count: int
    tool_call_count: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    first_ttft_ms: float | None


def run_harbor_task(
    client: ChatClient,
    root: Path,
    instruction: str,
    *,
    model: str,
    max_steps: int = 20,
    temperature: float = 0.0,
    seed: int = 42,
    session_id: str = "harbor-task",
) -> HarborRunSummary:
    """Run the single-agent harness in Harbor's task workspace."""

    repository = RepositoryTools(root)
    commands = CommandRunner(root)
    task = TaskSpec(
        task_id="harbor-task",
        instruction=instruction,
        source_dir=root,
        test_command=(
            "python",
            "-c",
            "print('Harbor runs independent verification')",
        ),
        timeout_seconds=30,
        tags=("harbor",),
    )
    dispatcher = ToolDispatcher(
        repository=repository,
        commands=commands,
        task=task,
    )
    result = SingleAgent(
        client,
        dispatcher,
        model=model,
        max_steps=max_steps,
        temperature=temperature,
        seed=seed,
        session_id=session_id,
    ).run(task)

    usages = [
        call.usage
        for call in result.model_calls
        if call.usage is not None
    ]
    costs = [
        usage.cost_usd
        for usage in usages
        if usage.cost_usd is not None
    ]

    return HarborRunSummary(
        status=result.status,
        final_answer=result.final_answer,
        error=result.error,
        steps=result.steps,
        wall_time_ms=result.wall_time_ms,
        model_call_count=len(result.model_calls),
        tool_call_count=len(result.tool_calls),
        input_tokens=(
            sum(usage.input_tokens for usage in usages)
            if usages
            else None
        ),
        output_tokens=(
            sum(usage.output_tokens for usage in usages)
            if usages
            else None
        ),
        cost_usd=sum(costs) if costs else None,
        first_ttft_ms=(
            result.model_calls[0].ttft_ms
            if result.model_calls
            else None
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Agent Harness Bench inside Harbor."
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--session-id",
        default="harbor-task",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required"
        )

    with OpenRouterClient(
        api_key,
        app_name="Agent Harness Bench - Harbor",
    ) as client:
        summary = run_harbor_task(
            client,
            arguments.root,
            arguments.instruction,
            model=arguments.model,
            max_steps=arguments.max_steps,
            temperature=arguments.temperature,
            seed=arguments.seed,
            session_id=arguments.session_id,
        )

    print(
        json.dumps(
            summary.model_dump(mode="json"),
            sort_keys=True,
        )
    )

    return 0 if summary.status is AgentStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())

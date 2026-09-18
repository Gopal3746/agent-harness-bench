from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_harness.llm import OpenRouterClient
from agent_harness.models import BenchmarkConfig, TaskSpec
from agent_harness.runner import SingleTaskRunner
from agent_harness.telemetry import JsonlRunWriter
from agent_harness.workspace import WorkspaceManager

PROGRAM_DESCRIPTION = (
    "Run and evaluate instrumented coding-agent workloads."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version="agent-harness 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run",
        help="run and verify one coding task",
    )
    run_parser.add_argument(
        "task",
        type=Path,
        help="path to a JSON task specification",
    )
    run_parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model identifier",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs.jsonl"),
        help="JSONL results path (default: results/runs.jsonl)",
    )
    run_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("workspaces"),
        help="isolated workspace directory (default: workspaces)",
    )
    run_parser.add_argument(
        "--run-id",
        help="optional stable run identifier",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="maximum model calls (default: 20)",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="model seed when supported (default: 42)",
    )
    run_parser.add_argument(
        "--retain-workspace",
        action="store_true",
        help="keep the workspace after the run",
    )

    return parser


def load_task(path: Path) -> TaskSpec:
    """Load a task, resolving its source directory by the JSON file."""

    resolved_path = path.expanduser().resolve()
    payload: Any = json.loads(
        resolved_path.read_text(encoding="utf-8")
    )

    if isinstance(payload, dict):
        source_dir = payload.get("source_dir")

        if isinstance(source_dir, str):
            source_path = Path(source_dir).expanduser()

            if not source_path.is_absolute():
                source_path = resolved_path.parent / source_path

            payload["source_dir"] = source_path.resolve()

    return TaskSpec.model_validate(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        parser.print_help()
        return 0

    if arguments.command == "run":
        try:
            return _run_task(arguments)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            ValidationError,
            ValueError,
        ) as error:
            parser.error(str(error))

    parser.error(f"unknown command: {arguments.command}")


def _run_task(arguments: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required"
        )

    task = load_task(arguments.task)
    config = BenchmarkConfig(
        model=arguments.model,
        repetitions=1,
        temperature=arguments.temperature,
        max_steps=arguments.max_steps,
        seed=arguments.seed,
    )
    workspaces = WorkspaceManager(
        arguments.workspace_root,
        retain_workspaces=arguments.retain_workspace,
    )

    with OpenRouterClient(
        api_key,
        app_name="Agent Harness Bench",
    ) as client:
        result = SingleTaskRunner(
            client,
            workspaces,
            config,
        ).run(task, run_id=arguments.run_id)

    record = JsonlRunWriter(arguments.output).append_result(result)

    print(
        json.dumps(
            {
                "agent_status": record.agent_status,
                "model": record.model,
                "output": str(arguments.output),
                "passed": record.passed,
                "run_id": record.run_id,
                "task_id": record.task_id,
                "wall_time_ms": record.wall_time_ms,
            },
            sort_keys=True,
        )
    )

    return 0 if record.passed else 1

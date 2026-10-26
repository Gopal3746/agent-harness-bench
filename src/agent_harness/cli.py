from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from agent_harness.analysis import (
    compare_strategies,
    load_run_records,
    summarize_records,
    write_comparison_csv,
    write_summary_csv,
)
from agent_harness.benchmark import BenchmarkRunner
from agent_harness.llm import OpenRouterClient
from agent_harness.models import (
    BenchmarkConfig,
    ExecutionStrategy,
)
from agent_harness.plotting import (
    plot_strategy_dashboard,
)
from agent_harness.runner import TaskRunner
from agent_harness.suite import BenchmarkSuiteRunner
from agent_harness.task_catalog import (
    discover_tasks,
    load_task,
)
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
        "--strategy",
        type=ExecutionStrategy,
        choices=tuple(ExecutionStrategy),
        default=ExecutionStrategy.SINGLE,
        help="orchestration strategy (default: single)",
    )
    run_parser.add_argument(
        "--parallelism",
        type=int,
        default=2,
        help="maximum parallel subagents (default: 2)",
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
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run a controlled multi-strategy experiment",
    )
    benchmark_parser.add_argument(
        "task",
        type=Path,
        help="path to a JSON task specification",
    )
    benchmark_parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model identifier",
    )
    benchmark_parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        type=ExecutionStrategy,
        choices=tuple(ExecutionStrategy),
        help=(
            "strategy to benchmark; repeat this option to select "
            "multiple strategies (default: all)"
        ),
    )
    benchmark_parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="runs per strategy (default: 5)",
    )
    benchmark_parser.add_argument(
        "--parallelism",
        type=int,
        default=2,
        help="maximum parallel subagents (default: 2)",
    )
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs.jsonl"),
        help="JSONL results path (default: results/runs.jsonl)",
    )
    benchmark_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("workspaces"),
        help="isolated workspace directory (default: workspaces)",
    )
    benchmark_parser.add_argument(
        "--experiment-id",
        help="optional stable experiment identifier",
    )
    benchmark_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="maximum model calls per agent (default: 20)",
    )
    benchmark_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="model seed when supported (default: 42)",
    )
    benchmark_parser.add_argument(
        "--retain-workspace",
        action="store_true",
        help="keep workspaces after each run",
    )
    suite_parser = subparsers.add_parser(
        "benchmark-suite",
        help="run every task in a benchmark catalog",
    )
    suite_parser.add_argument(
        "root",
        type=Path,
        help="directory containing benchmark task manifests",
    )
    suite_parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model identifier",
    )
    suite_parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        type=ExecutionStrategy,
        choices=tuple(ExecutionStrategy),
        help=(
            "strategy to benchmark; repeat this option "
            "to select multiple strategies (default: all)"
        ),
    )
    suite_parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="runs per task and strategy (default: 5)",
    )
    suite_parser.add_argument(
        "--parallelism",
        type=int,
        default=2,
        help="maximum parallel subagents (default: 2)",
    )
    suite_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/suite-runs.jsonl"),
        help=(
            "JSONL results path "
            "(default: results/suite-runs.jsonl)"
        ),
    )
    suite_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("workspaces"),
        help=(
            "isolated workspace directory "
            "(default: workspaces)"
        ),
    )
    suite_parser.add_argument(
        "--suite-id",
        help="optional stable suite identifier",
    )
    suite_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="maximum model calls per agent (default: 20)",
    )
    suite_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    suite_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="model seed when supported (default: 42)",
    )
    suite_parser.add_argument(
        "--retain-workspace",
        action="store_true",
        help="keep workspaces after each run",
    )
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="summarize benchmark JSONL by strategy",
    )
    analyze_parser.add_argument(
        "input",
        type=Path,
        help="path to benchmark JSONL records",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary.csv"),
        help="summary CSV path (default: results/summary.csv)",
    )
    analyze_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("results/comparisons.csv"),
        help=(
            "baseline comparison CSV path "
            "(default: results/comparisons.csv)"
        ),
    )
    analyze_parser.add_argument(
        "--baseline",
        type=ExecutionStrategy,
        choices=tuple(ExecutionStrategy),
        default=ExecutionStrategy.SINGLE,
        help="comparison baseline strategy (default: single)",
    )
    analyze_parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(
            "results/strategy-dashboard.png"
        ),
        help=(
            "strategy dashboard PNG path "
            "(default: results/strategy-dashboard.png)"
        ),
    )
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="list validated benchmark tasks",
    )
    tasks_parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("benchmarks/tasks"),
        help=(
            "benchmark task directory "
            "(default: benchmarks/tasks)"
        ),
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        parser.print_help()
        return 0

    try:
        if arguments.command == "run":
            return _run_task(arguments)

        if arguments.command == "benchmark":
            return _run_benchmark(arguments)

        if arguments.command == "benchmark-suite":
            return _run_benchmark_suite(arguments)

        if arguments.command == "tasks":
            return _list_tasks(arguments)

        if arguments.command == "analyze":
            return _analyze_runs(arguments)
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
        strategies=(arguments.strategy,),
        repetitions=1,
        parallelism=arguments.parallelism,
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
        result = TaskRunner(
            client,
            workspaces,
            config,
        ).run(
            task,
            strategy=arguments.strategy,
            run_id=arguments.run_id,
        )

    record = JsonlRunWriter(arguments.output).append_result(result)

    print(
        json.dumps(
            {
                "agent_status": record.agent_status,
                "model": record.model,
                "output": str(arguments.output),
                "passed": record.passed,
                "run_id": record.run_id,
                "strategy": record.strategy,
                "task_id": record.task_id,
                "wall_time_ms": record.wall_time_ms,
            },
            sort_keys=True,
        )
    )

    return 0 if record.passed else 1

def _run_benchmark(arguments: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required"
        )

    task = load_task(arguments.task)
    strategies = tuple(arguments.strategies or ExecutionStrategy)
    config = BenchmarkConfig(
        model=arguments.model,
        strategies=strategies,
        repetitions=arguments.repetitions,
        parallelism=arguments.parallelism,
        temperature=arguments.temperature,
        max_steps=arguments.max_steps,
        seed=arguments.seed,
    )
    workspaces = WorkspaceManager(
        arguments.workspace_root,
        retain_workspaces=arguments.retain_workspace,
    )
    writer = JsonlRunWriter(arguments.output)

    with OpenRouterClient(
        api_key,
        app_name="Agent Harness Bench",
    ) as client:
        result = BenchmarkRunner(
            client,
            workspaces,
            config,
            writer=writer,
        ).run(
            task,
            experiment_id=arguments.experiment_id,
        )

    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "model": config.model,
                "output": str(arguments.output),
                "passed_runs": result.passed_runs,
                "pass_rate": result.pass_rate,
                "strategies": [
                    strategy.value for strategy in config.strategies
                ],
                "task_id": result.task_id,
                "total_runs": result.total_runs,
            },
            sort_keys=True,
        )
    )

    return 0 if result.passed_runs == result.total_runs else 1

def _run_benchmark_suite(
    arguments: argparse.Namespace,
) -> int:
    api_key = os.environ.get(
        "OPENROUTER_API_KEY",
        "",
    )

    if not api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY environment variable "
            "is required"
        )

    tasks = discover_tasks(arguments.root)
    strategies = tuple(
        arguments.strategies or ExecutionStrategy
    )
    config = BenchmarkConfig(
        model=arguments.model,
        strategies=strategies,
        repetitions=arguments.repetitions,
        parallelism=arguments.parallelism,
        temperature=arguments.temperature,
        max_steps=arguments.max_steps,
        seed=arguments.seed,
    )
    workspaces = WorkspaceManager(
        arguments.workspace_root,
        retain_workspaces=arguments.retain_workspace,
    )
    writer = JsonlRunWriter(arguments.output)

    with OpenRouterClient(
        api_key,
        app_name="Agent Harness Bench",
    ) as client:
        benchmark_runner = BenchmarkRunner(
            client,
            workspaces,
            config,
            writer=writer,
        )
        result = BenchmarkSuiteRunner(
            benchmark_runner
        ).run(
            tasks,
            suite_id=arguments.suite_id,
        )

    print(
        json.dumps(
            {
                "model": config.model,
                "output": str(arguments.output),
                "pass_rate": result.pass_rate,
                "passed_runs": result.passed_runs,
                "strategies": [
                    strategy.value
                    for strategy in config.strategies
                ],
                "suite_id": result.suite_id,
                "task_count": result.total_tasks,
                "total_runs": result.total_runs,
            },
            sort_keys=True,
        )
    )

    return 0 if result.all_passed else 1

def _analyze_runs(arguments: argparse.Namespace) -> int:
    records = load_run_records(arguments.input)

    if not records:
        raise ValueError("no run records found")

    summaries = summarize_records(records)
    comparisons = compare_strategies(
        summaries,
        baseline=arguments.baseline,
    )

    write_summary_csv(arguments.output, summaries)
    write_comparison_csv(
        arguments.comparison_output,
        comparisons,
    )
    plot_strategy_dashboard(
        summaries,
        arguments.plot_output,
    )

    print(
        json.dumps(
            {
                "baseline": arguments.baseline,
                "comparison_output": str(
                    arguments.comparison_output
                ),
                "input": str(arguments.input),
                "output": str(arguments.output),
                "run_count": len(records),
                "strategies": [
                    summary.strategy.value
                    for summary in summaries
                ],
                "plot_output": str(
                    arguments.plot_output
                ),
            },
            sort_keys=True,
        )
    )

    return 0

def _list_tasks(arguments: argparse.Namespace) -> int:
    tasks = discover_tasks(arguments.root)

    print(
        json.dumps(
            {
                "root": str(
                    arguments.root.expanduser().resolve()
                ),
                "task_count": len(tasks),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "tags": task.tags,
                    }
                    for task in tasks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0

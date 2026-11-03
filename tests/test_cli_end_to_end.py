from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parents[1]
_TASK_ROOT = _PROJECT_ROOT / "benchmarks" / "tasks"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI through its Python module entrypoint."""

    environment = os.environ.copy()
    environment.pop("OPENROUTER_API_KEY", None)

    source_path = str(_PROJECT_ROOT / "src")
    existing_path = environment.get("PYTHONPATH", "")

    environment["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{existing_path}"
        if existing_path
        else source_path
    )

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_harness",
            *arguments,
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def parse_json_output(
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)

    assert isinstance(payload, dict)
    return payload


def test_cli_module_reports_version() -> None:
    result = run_cli("--version")

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "agent-harness 0.1.0"


def test_cli_module_lists_benchmark_catalog() -> None:
    result = run_cli(
        "tasks",
        str(_TASK_ROOT),
    )
    payload = parse_json_output(result)

    task_ids = [
        task["task_id"]
        for task in payload["tasks"]
    ]

    assert payload["task_count"] == 6
    assert task_ids == [
        "audit-events",
        "config-merge",
        "input-validation",
        "pagination-client",
        "service-retries",
        "ttl-cache-expiration",
    ]


def test_cli_module_plans_suite_without_api_key() -> None:
    result = run_cli(
        "benchmark-suite",
        str(_TASK_ROOT),
        "--model",
        "test/model",
        "--strategy",
        "single",
        "--strategy",
        "parallel",
        "--repetitions",
        "2",
        "--dry-run",
    )
    payload = parse_json_output(result)

    assert payload["dry_run"] is True
    assert payload["model"] == "test/model"
    assert payload["task_count"] == 6
    assert payload["strategies"] == [
        "single",
        "parallel",
    ]
    assert payload["repetitions"] == 2
    assert payload["planned_runs"] == 24
    assert payload["within_run_limit"] is True

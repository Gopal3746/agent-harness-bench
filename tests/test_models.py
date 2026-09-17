from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_harness import (
    BenchmarkConfig,
    ExecutionStrategy,
    TaskSpec,
)


def test_task_spec_normalizes_input() -> None:
    task = TaskSpec(
        task_id="fix-parser",
        instruction="  Fix the failing parser tests.  ",
        source_dir=Path("benchmarks/tasks/fix-parser"),
        test_command=["python", "-m", "pytest", "-q"],
        tags=["Parsing", "Bug-Fix"],
    )

    assert task.instruction == "Fix the failing parser tests."
    assert task.test_command == (
        "python",
        "-m",
        "pytest",
        "-q",
    )
    assert task.tags == ("parsing", "bug-fix")


@pytest.mark.parametrize(
    "task_id",
    [
        "",
        "Fix-Parser",
        "fix parser",
        "../fix-parser",
    ],
)
def test_task_spec_rejects_invalid_task_id(task_id: str) -> None:
    with pytest.raises(ValidationError):
        TaskSpec(
            task_id=task_id,
            instruction="Fix the parser.",
            source_dir=Path("tasks/parser"),
        )


def test_task_spec_rejects_blank_instruction() -> None:
    with pytest.raises(
        ValidationError,
        match="instruction must not be blank",
    ):
        TaskSpec(
            task_id="fix-parser",
            instruction="   ",
            source_dir=Path("tasks/parser"),
        )


def test_task_spec_rejects_empty_test_command() -> None:
    with pytest.raises(
        ValidationError,
        match="test_command must contain at least one argument",
    ):
        TaskSpec(
            task_id="fix-parser",
            instruction="Fix the parser.",
            source_dir=Path("tasks/parser"),
            test_command=[],
        )


def test_benchmark_config_parses_strategy_values() -> None:
    config = BenchmarkConfig(
        model="openai/example-model",
        strategies=["single", "parallel"],
        repetitions=5,
        parallelism=4,
    )

    assert config.strategies == (
        ExecutionStrategy.SINGLE,
        ExecutionStrategy.PARALLEL,
    )
    assert config.temperature == 0.0
    assert config.seed == 42


def test_parallel_strategy_requires_multiple_workers() -> None:
    with pytest.raises(
        ValidationError,
        match="parallelism must be at least 2",
    ):
        BenchmarkConfig(
            model="openai/example-model",
            strategies=["parallel"],
            parallelism=1,
        )


def test_benchmark_config_rejects_duplicate_strategies() -> None:
    with pytest.raises(
        ValidationError,
        match="strategies must be unique",
    ):
        BenchmarkConfig(
            model="openai/example-model",
            strategies=["single", "single"],
        )


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            model="openai/example-model",
            unknown_setting=True,
        )

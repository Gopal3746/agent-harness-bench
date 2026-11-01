from pathlib import Path

import pytest

from agent_harness import (
    prepare_output_file,
    validate_run_budget,
)


def test_run_budget_accepts_plan_at_limit() -> None:
    validate_run_budget(12, 12)


def test_run_budget_rejects_plan_over_limit() -> None:
    with pytest.raises(
        ValueError,
        match="planned run count 13 exceeds the safety limit 12",
    ):
        validate_run_budget(13, 12)


@pytest.mark.parametrize(
    ("planned_runs", "max_total_runs", "message"),
    [
        (0, 10, "planned_runs"),
        (1, 0, "max_total_runs"),
    ],
)
def test_run_budget_rejects_non_positive_values(
    planned_runs: int,
    max_total_runs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_run_budget(
            planned_runs,
            max_total_runs,
        )


def test_output_guard_accepts_missing_file(
    tmp_path: Path,
) -> None:
    prepare_output_file(tmp_path / "runs.jsonl")


def test_output_guard_accepts_empty_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs.jsonl"
    output.touch()

    prepare_output_file(output)

    assert output.exists()


def test_output_guard_rejects_non_empty_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs.jsonl"
    output.write_text("existing result\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="already contains data",
    ):
        prepare_output_file(output)

    assert output.read_text(encoding="utf-8") == (
        "existing result\n"
    )


def test_output_guard_removes_file_when_overwriting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs.jsonl"
    output.write_text("existing result\n", encoding="utf-8")

    prepare_output_file(output, overwrite=True)

    assert not output.exists()


def test_output_guard_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="output path is not a file",
    ):
        prepare_output_file(tmp_path)

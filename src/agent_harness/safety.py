from __future__ import annotations

from pathlib import Path

DEFAULT_MAX_TOTAL_RUNS = 100


def validate_run_budget(
    planned_runs: int,
    max_total_runs: int,
) -> None:
    """Reject invalid or unexpectedly large experiments."""

    if planned_runs < 1:
        raise ValueError("planned_runs must be at least 1")

    if max_total_runs < 1:
        raise ValueError("max_total_runs must be at least 1")

    if planned_runs > max_total_runs:
        raise ValueError(
            f"planned run count {planned_runs} exceeds "
            f"the safety limit {max_total_runs}; reduce the "
            "experiment or pass a larger --max-total-runs value"
        )


def prepare_output_file(
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Protect existing benchmark results from accidental mixing."""

    resolved_path = path.expanduser()

    if not resolved_path.exists():
        return

    if not resolved_path.is_file():
        raise ValueError(
            f"output path is not a file: {resolved_path}"
        )

    if resolved_path.stat().st_size == 0:
        return

    if not overwrite:
        raise ValueError(
            f"output file already contains data: {resolved_path}; "
            "choose a new path or pass --overwrite-output"
        )

    resolved_path.unlink()

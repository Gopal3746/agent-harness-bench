from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median

from pydantic import ValidationError

from agent_harness.models import ExecutionStrategy
from agent_harness.telemetry import RunRecord


@dataclass(frozen=True, slots=True)
class StrategySummary:
    """Aggregate workload and correctness metrics for one strategy."""

    strategy: ExecutionStrategy
    run_count: int
    passed_count: int
    pass_rate: float
    mean_wall_time_ms: float
    median_wall_time_ms: float
    mean_ttft_ms: float | None
    mean_output_tokens_per_second: float | None
    mean_model_calls: float
    mean_tool_calls: float
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    total_cost_usd: float | None


def load_run_records(path: Path) -> tuple[RunRecord, ...]:
    """Load and validate newline-delimited benchmark records."""

    records: list[RunRecord] = []

    with path.expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                records.append(RunRecord.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(
                    f"invalid run record at line {line_number}: {error}"
                ) from error

    return tuple(records)


def summarize_records(
    records: Sequence[RunRecord],
) -> tuple[StrategySummary, ...]:
    """Aggregate benchmark metrics by orchestration strategy."""

    grouped: dict[
        ExecutionStrategy,
        list[RunRecord],
    ] = defaultdict(list)

    for record in records:
        grouped[record.strategy].append(record)

    return tuple(
        _summarize_strategy(strategy, grouped[strategy])
        for strategy in ExecutionStrategy
        if strategy in grouped
    )

def write_summary_csv(
    path: Path,
    summaries: Sequence[StrategySummary],
) -> None:
    """Write per-strategy summaries to a CSV file."""

    output = path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(StrategySummary.__dataclass_fields__)

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in summaries:
            row = asdict(summary)
            row["strategy"] = summary.strategy.value
            writer.writerow(row)

def _summarize_strategy(
    strategy: ExecutionStrategy,
    records: Sequence[RunRecord],
) -> StrategySummary:
    wall_times = [record.wall_time_ms for record in records]
    reported_costs = [
        record.total_cost_usd
        for record in records
        if record.total_cost_usd is not None
    ]
    passed_count = sum(record.passed for record in records)

    return StrategySummary(
        strategy=strategy,
        run_count=len(records),
        passed_count=passed_count,
        pass_rate=passed_count / len(records),
        mean_wall_time_ms=fmean(wall_times),
        median_wall_time_ms=median(wall_times),
        mean_ttft_ms=_optional_mean(
            record.mean_ttft_ms for record in records
        ),
        mean_output_tokens_per_second=_optional_mean(
            record.output_tokens_per_second for record in records
        ),
        mean_model_calls=fmean(
            record.model_call_count for record in records
        ),
        mean_tool_calls=fmean(
            record.tool_call_count for record in records
        ),
        mean_input_tokens=_optional_mean(
            record.total_input_tokens for record in records
        ),
        mean_output_tokens=_optional_mean(
            record.total_output_tokens for record in records
        ),
        total_cost_usd=(
            sum(reported_costs) if reported_costs else None
        ),
    )


def _optional_mean(
    values: Iterable[float | int | None],
) -> float | None:
    reported = [value for value in values if value is not None]
    return fmean(reported) if reported else None

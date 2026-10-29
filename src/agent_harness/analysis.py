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

@dataclass(frozen=True, slots=True)
class TaskStrategySummary:
    """Aggregate metrics for one task and strategy."""

    task_id: str
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

def summarize_records_by_task(
    records: Sequence[RunRecord],
) -> tuple[TaskStrategySummary, ...]:
    """Aggregate metrics by task and orchestration strategy."""

    grouped: dict[
        tuple[str, ExecutionStrategy],
        list[RunRecord],
    ] = defaultdict(list)

    for record in records:
        key = (record.task_id, record.strategy)
        grouped[key].append(record)

    task_ids = sorted(
        {record.task_id for record in records}
    )

    return tuple(
        _summarize_task_strategy(
            task_id,
            strategy,
            grouped[(task_id, strategy)],
        )
        for task_id in task_ids
        for strategy in ExecutionStrategy
        if (task_id, strategy) in grouped
    )

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

def write_task_summary_csv(
    path: Path,
    summaries: Sequence[TaskStrategySummary],
) -> None:
    """Write per-task strategy summaries to CSV."""

    output = path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(
        TaskStrategySummary.__dataclass_fields__
    )

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

def compare_strategies(
    summaries: Sequence[StrategySummary],
    *,
    baseline: ExecutionStrategy = ExecutionStrategy.SINGLE,
) -> tuple[StrategyComparison, ...]:
    """Compare every non-baseline strategy with the baseline."""

    by_strategy = {
        summary.strategy: summary
        for summary in summaries
    }

    try:
        baseline_summary = by_strategy[baseline]
    except KeyError as error:
        raise ValueError(
            f"baseline strategy {baseline.value!r} is missing"
        ) from error

    return tuple(
        _compare_summary(baseline_summary, summary)
        for summary in summaries
        if summary.strategy is not baseline
    )


def write_comparison_csv(
    path: Path,
    comparisons: Sequence[StrategyComparison],
) -> None:
    """Write baseline-relative strategy comparisons to CSV."""

    output = path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(
        StrategyComparison.__dataclass_fields__
    )

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

        for comparison in comparisons:
            row = asdict(comparison)
            row["baseline_strategy"] = (
                comparison.baseline_strategy.value
            )
            row["strategy"] = comparison.strategy.value
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

def _summarize_task_strategy(
    task_id: str,
    strategy: ExecutionStrategy,
    records: Sequence[RunRecord],
) -> TaskStrategySummary:
    summary = _summarize_strategy(
        strategy,
        records,
    )

    return TaskStrategySummary(
        task_id=task_id,
        strategy=summary.strategy,
        run_count=summary.run_count,
        passed_count=summary.passed_count,
        pass_rate=summary.pass_rate,
        mean_wall_time_ms=summary.mean_wall_time_ms,
        median_wall_time_ms=summary.median_wall_time_ms,
        mean_ttft_ms=summary.mean_ttft_ms,
        mean_output_tokens_per_second=(
            summary.mean_output_tokens_per_second
        ),
        mean_model_calls=summary.mean_model_calls,
        mean_tool_calls=summary.mean_tool_calls,
        mean_input_tokens=summary.mean_input_tokens,
        mean_output_tokens=summary.mean_output_tokens,
        total_cost_usd=summary.total_cost_usd,
    )

@dataclass(frozen=True, slots=True)
class StrategyComparison:
    """Relative metrics for one strategy against a baseline."""

    baseline_strategy: ExecutionStrategy
    strategy: ExecutionStrategy
    wall_time_speedup: float | None
    wall_time_change_pct: float | None
    pass_rate_change_pp: float
    input_tokens_change_pct: float | None
    output_tokens_change_pct: float | None
    cost_per_run_change_pct: float | None

def _compare_summary(
    baseline: StrategySummary,
    candidate: StrategySummary,
) -> StrategyComparison:
    baseline_cost_per_run = _cost_per_run(baseline)
    candidate_cost_per_run = _cost_per_run(candidate)

    return StrategyComparison(
        baseline_strategy=baseline.strategy,
        strategy=candidate.strategy,
        wall_time_speedup=_safe_ratio(
            baseline.mean_wall_time_ms,
            candidate.mean_wall_time_ms,
        ),
        wall_time_change_pct=_percent_change(
            candidate.mean_wall_time_ms,
            baseline.mean_wall_time_ms,
        ),
        pass_rate_change_pp=(
            candidate.pass_rate - baseline.pass_rate
        )
        * 100,
        input_tokens_change_pct=_percent_change(
            candidate.mean_input_tokens,
            baseline.mean_input_tokens,
        ),
        output_tokens_change_pct=_percent_change(
            candidate.mean_output_tokens,
            baseline.mean_output_tokens,
        ),
        cost_per_run_change_pct=_percent_change(
            candidate_cost_per_run,
            baseline_cost_per_run,
        ),
    )


def _cost_per_run(
    summary: StrategySummary,
) -> float | None:
    if summary.total_cost_usd is None:
        return None

    return summary.total_cost_usd / summary.run_count


def _safe_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None

    return numerator / denominator


def _percent_change(
    value: float | None,
    baseline: float | None,
) -> float | None:
    if value is None or baseline in (None, 0):
        return None

    return ((value - baseline) / baseline) * 100

def _optional_mean(
    values: Iterable[float | int | None],
) -> float | None:
    reported = [value for value in values if value is not None]
    return fmean(reported) if reported else None

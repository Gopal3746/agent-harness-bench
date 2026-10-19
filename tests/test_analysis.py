from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_harness import (
    AgentStatus,
    ExecutionStrategy,
    JsonlRunWriter,
    RunRecord,
    VerificationMetrics,
    load_run_records,
    summarize_records,
)


def make_record(
    run_id: str,
    strategy: ExecutionStrategy,
    *,
    passed: bool,
    wall_time_ms: float,
    ttft_ms: float | None,
    throughput: float | None,
    model_calls: int,
    tool_calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
) -> RunRecord:
    return RunRecord(
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_id=run_id,
        task_id="sample-task",
        strategy=strategy,
        model="test/model",
        passed=passed,
        agent_status=AgentStatus.COMPLETED,
        agent_error=None,
        final_answer="Done.",
        steps=model_calls,
        wall_time_ms=wall_time_ms,
        model_call_count=model_calls,
        tool_call_count=tool_calls,
        tool_error_count=0,
        usage_reported_call_count=(
            model_calls if input_tokens is not None else 0
        ),
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cost_usd=cost_usd,
        total_model_latency_ms=wall_time_ms / 2,
        total_tool_duration_ms=10.0,
        first_ttft_ms=ttft_ms,
        mean_ttft_ms=ttft_ms,
        mean_stream_event_interval_ms=5.0,
        output_tokens_per_second=throughput,
        model_calls=(),
        tool_calls=(),
        verification=VerificationMetrics(
            command=("pytest", "-q"),
            succeeded=passed,
            exit_code=0 if passed else 1,
            timed_out=False,
            duration_ms=25.0,
            output_truncated=False,
            stdout="",
            stderr="",
        ),
    )


def test_summarize_records_groups_metrics_by_strategy() -> None:
    records = (
        make_record(
            "single-1",
            ExecutionStrategy.SINGLE,
            passed=True,
            wall_time_ms=100.0,
            ttft_ms=10.0,
            throughput=40.0,
            model_calls=1,
            tool_calls=2,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
        ),
        make_record(
            "single-2",
            ExecutionStrategy.SINGLE,
            passed=False,
            wall_time_ms=300.0,
            ttft_ms=30.0,
            throughput=60.0,
            model_calls=3,
            tool_calls=4,
            input_tokens=300,
            output_tokens=60,
            cost_usd=0.03,
        ),
        make_record(
            "parallel-1",
            ExecutionStrategy.PARALLEL,
            passed=True,
            wall_time_ms=80.0,
            ttft_ms=None,
            throughput=None,
            model_calls=2,
            tool_calls=1,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
        ),
    )

    single, parallel = summarize_records(records)

    assert single.strategy is ExecutionStrategy.SINGLE
    assert single.run_count == 2
    assert single.passed_count == 1
    assert single.pass_rate == 0.5
    assert single.mean_wall_time_ms == 200.0
    assert single.median_wall_time_ms == 200.0
    assert single.mean_ttft_ms == 20.0
    assert single.mean_output_tokens_per_second == 50.0
    assert single.mean_model_calls == 2.0
    assert single.mean_tool_calls == 3.0
    assert single.mean_input_tokens == 200.0
    assert single.mean_output_tokens == 40.0
    assert single.total_cost_usd == pytest.approx(0.04)

    assert parallel.strategy is ExecutionStrategy.PARALLEL
    assert parallel.run_count == 1
    assert parallel.pass_rate == 1.0
    assert parallel.mean_wall_time_ms == 80.0
    assert parallel.mean_ttft_ms is None
    assert parallel.mean_output_tokens_per_second is None
    assert parallel.total_cost_usd is None


def test_load_run_records_reads_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "runs.jsonl"
    writer = JsonlRunWriter(output)
    expected = make_record(
        "single-1",
        ExecutionStrategy.SINGLE,
        passed=True,
        wall_time_ms=100.0,
        ttft_ms=10.0,
        throughput=40.0,
        model_calls=1,
        tool_calls=2,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.01,
    )
    writer.append(expected)

    assert load_run_records(output) == (expected,)


def test_load_run_records_reports_invalid_line(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs.jsonl"
    output.write_text(
        "\n{not valid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_run_records(output)

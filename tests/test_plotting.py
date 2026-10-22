from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness import (
    ExecutionStrategy,
    StrategySummary,
    plot_strategy_dashboard,
)


def test_plot_strategy_dashboard_writes_png(
    tmp_path: Path,
) -> None:
    summaries = (
        StrategySummary(
            strategy=ExecutionStrategy.SINGLE,
            run_count=2,
            passed_count=1,
            pass_rate=0.5,
            mean_wall_time_ms=200.0,
            median_wall_time_ms=190.0,
            mean_ttft_ms=20.0,
            mean_output_tokens_per_second=50.0,
            mean_model_calls=2.0,
            mean_tool_calls=3.0,
            mean_input_tokens=200.0,
            mean_output_tokens=40.0,
            total_cost_usd=0.04,
        ),
        StrategySummary(
            strategy=ExecutionStrategy.PARALLEL,
            run_count=2,
            passed_count=2,
            pass_rate=1.0,
            mean_wall_time_ms=100.0,
            median_wall_time_ms=95.0,
            mean_ttft_ms=None,
            mean_output_tokens_per_second=None,
            mean_model_calls=3.0,
            mean_tool_calls=4.0,
            mean_input_tokens=None,
            mean_output_tokens=None,
            total_cost_usd=None,
        ),
    )
    output = tmp_path / "plots" / "dashboard.png"

    result = plot_strategy_dashboard(
        summaries,
        output,
    )

    assert result == output
    assert output.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_plot_strategy_dashboard_rejects_empty_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        plot_strategy_dashboard(
            (),
            tmp_path / "dashboard.png",
        )

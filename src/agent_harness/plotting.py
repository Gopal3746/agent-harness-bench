from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from agent_harness.analysis import StrategySummary


def plot_strategy_dashboard(
    summaries: Sequence[StrategySummary],
    output: Path,
) -> Path:
    """Render a dashboard comparing strategy-level metrics."""

    if not summaries:
        raise ValueError(
            "at least one strategy summary is required"
        )

    destination = output.expanduser()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    labels = [
        summary.strategy.value
        for summary in summaries
    ]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(15, 8.5),
            constrained_layout=True,
        )
        figure.suptitle(
            "Agent Harness Strategy Comparison",
            fontsize=16,
            fontweight="bold",
        )

        _plot_latency(axes[0, 0], labels, summaries)
        _plot_pass_rate(axes[0, 1], labels, summaries)

        _plot_optional_metric(
            axes[0, 2],
            labels,
            [
                summary.mean_ttft_ms
                for summary in summaries
            ],
            title="Mean TTFT",
            ylabel="Milliseconds",
            color="#f59e0b",
        )
        _plot_optional_metric(
            axes[1, 0],
            labels,
            [
                summary.mean_output_tokens_per_second
                for summary in summaries
            ],
            title="Generation Throughput",
            ylabel="Output tokens / second",
            color="#8b5cf6",
        )

        _plot_tokens(axes[1, 1], labels, summaries)

        _plot_optional_metric(
            axes[1, 2],
            labels,
            [
                _cost_per_run(summary)
                for summary in summaries
            ],
            title="Mean Cost per Run",
            ylabel="USD",
            color="#ef4444",
        )

        figure.savefig(
            destination,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    return destination


def _plot_latency(
    axis: Axes,
    labels: Sequence[str],
    summaries: Sequence[StrategySummary],
) -> None:
    positions = list(range(len(summaries)))
    width = 0.36

    mean_positions = [
        position - width / 2
        for position in positions
    ]
    median_positions = [
        position + width / 2
        for position in positions
    ]

    axis.bar(
        mean_positions,
        [
            summary.mean_wall_time_ms
            for summary in summaries
        ],
        width,
        label="Mean",
        color="#2563eb",
    )
    axis.bar(
        median_positions,
        [
            summary.median_wall_time_ms
            for summary in summaries
        ],
        width,
        label="Median",
        color="#60a5fa",
    )
    axis.set_title("Task Wall Time")
    axis.set_ylabel("Milliseconds")
    axis.set_xticks(positions, labels)
    axis.legend()


def _plot_pass_rate(
    axis: Axes,
    labels: Sequence[str],
    summaries: Sequence[StrategySummary],
) -> None:
    axis.bar(
        labels,
        [
            summary.pass_rate * 100
            for summary in summaries
        ],
        color="#10b981",
    )
    axis.set_title("Task Success Rate")
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 100)


def _plot_tokens(
    axis: Axes,
    labels: Sequence[str],
    summaries: Sequence[StrategySummary],
) -> None:
    inputs = [
        summary.mean_input_tokens
        for summary in summaries
    ]
    outputs = [
        summary.mean_output_tokens
        for summary in summaries
    ]

    if all(
        value is None
        for value in (*inputs, *outputs)
    ):
        _show_no_data(axis, "Mean Token Usage")
        return

    positions = list(range(len(summaries)))
    width = 0.36

    input_positions = [
        position - width / 2
        for position in positions
    ]
    output_positions = [
        position + width / 2
        for position in positions
    ]

    axis.bar(
        input_positions,
        _plot_values(inputs),
        width,
        label="Input",
        color="#0f766e",
    )
    axis.bar(
        output_positions,
        _plot_values(outputs),
        width,
        label="Output",
        color="#5eead4",
    )
    axis.set_title("Mean Token Usage")
    axis.set_ylabel("Tokens per run")
    axis.set_xticks(positions, labels)
    axis.legend()


def _plot_optional_metric(
    axis: Axes,
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    if all(value is None for value in values):
        _show_no_data(axis, title)
        return

    axis.bar(
        labels,
        _plot_values(values),
        color=color,
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)


def _show_no_data(
    axis: Axes,
    title: str,
) -> None:
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        "No data reported",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_values(
    values: Sequence[float | None],
) -> list[float]:
    return [
        float("nan") if value is None else value
        for value in values
    ]


def _cost_per_run(
    summary: StrategySummary,
) -> float | None:
    if summary.total_cost_usd is None:
        return None

    return summary.total_cost_usd / summary.run_count

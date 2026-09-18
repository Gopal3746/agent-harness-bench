from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent_harness.agent import AgentStatus
from agent_harness.llm import ChatCompletionResult
from agent_harness.models import ExecutionStrategy
from agent_harness.runner import TaskRunResult


class ModelCallMetrics(BaseModel):
    """Inference metrics for one model request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int
    request_id: str | None
    generation_id: str | None
    model: str | None
    finish_reason: str | None
    ttft_ms: float | None
    total_latency_ms: float
    generation_duration_ms: float | None
    mean_stream_event_interval_ms: float | None
    output_tokens_per_second: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    stream_event_count: int
    raw_event_count: int
    tool_call_count: int


class ToolCallMetrics(BaseModel):
    """Timing and outcome for one coding tool invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    name: str
    is_error: bool
    duration_ms: float


class VerificationMetrics(BaseModel):
    """Independent test result for one completed workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: tuple[str, ...]
    succeeded: bool
    exit_code: int | None
    timed_out: bool
    duration_ms: float
    output_truncated: bool
    stdout: str
    stderr: str


class RunRecord(BaseModel):
    """Serializable metrics and outcome for one benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    recorded_at: datetime
    run_id: str
    task_id: str
    strategy: ExecutionStrategy
    model: str
    passed: bool
    agent_status: AgentStatus
    agent_error: str | None
    final_answer: str
    steps: int
    wall_time_ms: float
    model_call_count: int
    tool_call_count: int
    tool_error_count: int
    usage_reported_call_count: int
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_cost_usd: float | None
    total_model_latency_ms: float
    total_tool_duration_ms: float
    first_ttft_ms: float | None
    mean_ttft_ms: float | None
    mean_stream_event_interval_ms: float | None
    output_tokens_per_second: float | None
    model_calls: tuple[ModelCallMetrics, ...]
    tool_calls: tuple[ToolCallMetrics, ...]
    verification: VerificationMetrics

    @classmethod
    def from_result(
        cls,
        result: TaskRunResult,
        *,
        recorded_at: datetime | None = None,
    ) -> RunRecord:
        """Build a stable telemetry record from a task result."""

        calls = result.agent.model_calls
        usages = [
            call.usage
            for call in calls
            if call.usage is not None
        ]
        ttft_values = [
            call.ttft_ms
            for call in calls
            if call.ttft_ms is not None
        ]
        stream_intervals = [
            interval
            for call in calls
            for interval in call.stream_event_intervals_ms
        ]
        generation_duration_ms = sum(
            call.generation_duration_ms or 0.0
            for call in calls
        )
        total_output_tokens = (
            sum(usage.output_tokens for usage in usages)
            if usages
            else None
        )
        reported_costs = [
            usage.cost_usd
            for usage in usages
            if usage.cost_usd is not None
        ]

        return cls(
            recorded_at=recorded_at or datetime.now(UTC),
            run_id=result.run_id,
            task_id=result.task_id,
            strategy=result.strategy,
            model=result.model,
            passed=result.passed,
            agent_status=result.agent.status,
            agent_error=result.agent.error,
            final_answer=result.agent.final_answer,
            steps=result.agent.steps,
            wall_time_ms=result.wall_time_ms,
            model_call_count=len(calls),
            tool_call_count=len(result.agent.tool_calls),
            tool_error_count=sum(
                call.is_error
                for call in result.agent.tool_calls
            ),
            usage_reported_call_count=len(usages),
            total_input_tokens=(
                sum(usage.input_tokens for usage in usages)
                if usages
                else None
            ),
            total_output_tokens=total_output_tokens,
            total_cost_usd=(
                sum(reported_costs)
                if reported_costs
                else None
            ),
            total_model_latency_ms=sum(
                call.total_latency_ms for call in calls
            ),
            total_tool_duration_ms=sum(
                call.duration_ms
                for call in result.agent.tool_calls
            ),
            first_ttft_ms=(calls[0].ttft_ms if calls else None),
            mean_ttft_ms=(
                fmean(ttft_values) if ttft_values else None
            ),
            mean_stream_event_interval_ms=(
                fmean(stream_intervals)
                if stream_intervals
                else None
            ),
            output_tokens_per_second=(
                total_output_tokens
                / (generation_duration_ms / 1_000)
                if total_output_tokens is not None
                and generation_duration_ms > 0
                else None
            ),
            model_calls=tuple(
                _model_call_metrics(index, call)
                for index, call in enumerate(calls, start=1)
            ),
            tool_calls=tuple(
                ToolCallMetrics(
                    step=call.step,
                    name=call.name,
                    is_error=call.is_error,
                    duration_ms=call.duration_ms,
                )
                for call in result.agent.tool_calls
            ),
            verification=VerificationMetrics(
                command=result.verification.command,
                succeeded=result.verification.succeeded,
                exit_code=result.verification.exit_code,
                timed_out=result.verification.timed_out,
                duration_ms=result.verification.duration_ms,
                output_truncated=(
                    result.verification.output_truncated
                ),
                stdout=result.verification.stdout,
                stderr=result.verification.stderr,
            ),
        )


class JsonlRunWriter:
    """Appends benchmark records as newline-delimited JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self._lock = Lock()

    def append(self, record: RunRecord) -> None:
        """Append exactly one JSON object and flush it to disk."""

        line = record.model_dump_json()

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
                output.write("\n")
                output.flush()

    def append_result(
        self,
        result: TaskRunResult,
        *,
        recorded_at: datetime | None = None,
    ) -> RunRecord:
        """Convert and append a task result, returning its record."""

        record = RunRecord.from_result(
            result,
            recorded_at=recorded_at,
        )
        self.append(record)
        return record


def _model_call_metrics(
    index: int,
    call: ChatCompletionResult,
) -> ModelCallMetrics:
    usage = call.usage

    return ModelCallMetrics(
        call_index=index,
        request_id=call.request_id,
        generation_id=call.generation_id,
        model=call.model,
        finish_reason=call.finish_reason,
        ttft_ms=call.ttft_ms,
        total_latency_ms=call.total_latency_ms,
        generation_duration_ms=call.generation_duration_ms,
        mean_stream_event_interval_ms=(
            call.mean_stream_event_interval_ms
        ),
        output_tokens_per_second=call.output_tokens_per_second,
        input_tokens=(usage.input_tokens if usage else None),
        output_tokens=(usage.output_tokens if usage else None),
        cached_tokens=(usage.cached_tokens if usage else None),
        reasoning_tokens=(usage.reasoning_tokens if usage else None),
        cost_usd=(usage.cost_usd if usage else None),
        stream_event_count=len(call.stream_event_timestamps_ns),
        raw_event_count=call.raw_event_count,
        tool_call_count=len(call.tool_calls),
    )

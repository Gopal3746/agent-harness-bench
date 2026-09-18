from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from agent_harness import (
    AgentRunResult,
    AgentStatus,
    ChatCompletionResult,
    CommandResult,
    ExecutionStrategy,
    JsonlRunWriter,
    RunRecord,
    StreamedToolCall,
    TaskRunResult,
    TaskWorkspace,
    TokenUsage,
    ToolCallRecord,
)


def make_task_result(tmp_path: Path) -> TaskRunResult:
    model_call = ChatCompletionResult(
        request_id="request-1",
        generation_id="generation-1",
        model="test/model",
        content="Implemented and tested.",
        reasoning="",
        tool_calls=(
            StreamedToolCall(
                index=0,
                call_id="call-1",
                name="run_tests",
                arguments="{}",
            ),
        ),
        finish_reason="tool_calls",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_tokens=10,
            reasoning_tokens=5,
            cost_usd=0.002,
        ),
        request_started_ns=1_000_000_000,
        first_output_ns=1_100_000_000,
        last_output_ns=1_300_000_000,
        request_finished_ns=1_350_000_000,
        stream_event_timestamps_ns=(
            1_100_000_000,
            1_200_000_000,
            1_300_000_000,
        ),
        raw_event_count=4,
    )
    agent = AgentRunResult(
        task_id="sample-task",
        status=AgentStatus.COMPLETED,
        final_answer="Implemented and tested.",
        error=None,
        steps=1,
        wall_time_ms=400.0,
        messages=(),
        model_calls=(model_call,),
        tool_calls=(
            ToolCallRecord(
                step=1,
                call_id="call-1",
                name="run_tests",
                arguments="{}",
                response='{"succeeded": true}',
                is_error=False,
                duration_ms=25.0,
            ),
        ),
    )
    verification = CommandResult(
        command=("pytest", "-q"),
        exit_code=0,
        stdout="4 passed\n",
        stderr="",
        duration_ms=50.0,
        timed_out=False,
        output_truncated=False,
    )

    return TaskRunResult(
        run_id="run-001",
        task_id="sample-task",
        strategy=ExecutionStrategy.SINGLE,
        model="test/model",
        workspace=TaskWorkspace(
            task_id="sample-task",
            run_id="run-001",
            path=tmp_path / "workspace",
        ),
        agent=agent,
        verification=verification,
        wall_time_ms=500.0,
    )


def test_run_record_calculates_workload_metrics(
    tmp_path: Path,
) -> None:
    recorded_at = datetime(
        2026,
        9,
        18,
        12,
        0,
        tzinfo=UTC,
    )

    record = RunRecord.from_result(
        make_task_result(tmp_path),
        recorded_at=recorded_at,
    )

    assert record.schema_version == 1
    assert record.recorded_at == recorded_at
    assert record.passed is True
    assert record.model_call_count == 1
    assert record.tool_call_count == 1
    assert record.tool_error_count == 0
    assert record.total_input_tokens == 100
    assert record.total_output_tokens == 20
    assert record.total_cost_usd == 0.002
    assert record.first_ttft_ms == 100.0
    assert record.mean_ttft_ms == 100.0
    assert record.mean_stream_event_interval_ms == 100.0
    assert record.output_tokens_per_second == 100.0
    assert record.model_calls[0].raw_event_count == 4
    assert record.verification.succeeded is True


def test_missing_usage_remains_unknown(
    tmp_path: Path,
) -> None:
    result = make_task_result(tmp_path)
    call_without_usage = replace(
        result.agent.model_calls[0],
        usage=None,
    )
    result_without_usage = replace(
        result,
        agent=replace(
            result.agent,
            model_calls=(call_without_usage,),
        ),
    )

    record = RunRecord.from_result(result_without_usage)

    assert record.usage_reported_call_count == 0
    assert record.total_input_tokens is None
    assert record.total_output_tokens is None
    assert record.total_cost_usd is None
    assert record.output_tokens_per_second is None


def test_jsonl_writer_appends_one_object_per_line(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results" / "runs.jsonl"
    writer = JsonlRunWriter(output_path)
    result = make_task_result(tmp_path)

    writer.append_result(result)
    writer.append_result(result)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["run_id"] == "run-001"
    assert first["strategy"] == "single"
    assert first["verification"]["succeeded"] is True
    assert second["run_id"] == "run-001"

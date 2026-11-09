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

    assert record.schema_version == 2
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
    assert record.tool_calls[0].call_id == "call-1"
    assert record.tool_calls[0].arguments == "{}"
    assert record.tool_calls[0].response == '{"succeeded":true}'
    assert record.tool_calls[0].arguments_truncated is False
    assert record.tool_calls[0].response_truncated is False
    assert record.verification.succeeded is True


def test_tool_traces_are_redacted_and_bounded(
    tmp_path: Path,
) -> None:
    result = make_task_result(tmp_path)
    secret = "sk-or-v1-this-must-not-appear"
    long_text = "x" * 4_000
    traced_call = replace(
        result.agent.tool_calls[0],
        arguments=json.dumps(
            {
                "api_key": secret,
                "content": long_text,
            }
        ),
        response=json.dumps(
            {
                "access_token": secret,
                "stdout": (
                    f"Authorization: Bearer {secret}\n"
                    f"{long_text}"
                ),
            }
        ),
    )
    traced_result = replace(
        result,
        agent=replace(
            result.agent,
            tool_calls=(traced_call,),
        ),
    )

    record = RunRecord.from_result(traced_result)
    trace = record.tool_calls[0]

    assert trace.arguments is not None
    assert trace.response is not None
    assert len(trace.arguments) <= 2_000
    assert len(trace.response) <= 2_000
    assert trace.arguments_truncated is True
    assert trace.response_truncated is True
    assert secret not in trace.arguments
    assert secret not in trace.response
    assert "[REDACTED]" in trace.arguments
    assert "[REDACTED]" in trace.response


def test_version_one_records_remain_loadable(
    tmp_path: Path,
) -> None:
    payload = RunRecord.from_result(
        make_task_result(tmp_path)
    ).model_dump(mode="json")
    payload["schema_version"] = 1

    for tool_call in payload["tool_calls"]:
        tool_call.pop("call_id")
        tool_call.pop("arguments")
        tool_call.pop("response")
        tool_call.pop("arguments_truncated")
        tool_call.pop("response_truncated")

    record = RunRecord.model_validate(payload)

    assert record.schema_version == 1
    assert record.tool_calls[0].arguments is None
    assert record.tool_calls[0].response is None


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
    assert first["schema_version"] == 2
    assert first["strategy"] == "single"
    assert first["tool_calls"][0]["arguments"] == "{}"
    assert first["tool_calls"][0]["response"] == (
        '{"succeeded":true}'
    )
    assert first["verification"]["succeeded"] is True
    assert second["run_id"] == "run-001"

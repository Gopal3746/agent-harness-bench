from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_harness import (
    ChatCompletionResult,
    StreamedToolCall,
    TokenUsage,
)
from agent_harness.integrations import run_harbor_task


class FakeChatClient:
    def __init__(
        self,
        responses: list[ChatCompletionResult],
    ) -> None:
        self.responses = responses

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        session_id: str | None = None,
    ) -> ChatCompletionResult:
        del (
            messages,
            model,
            tools,
            temperature,
            max_tokens,
            seed,
            session_id,
        )
        return self.responses.pop(0)


def make_completion(
    *,
    content: str = "",
    tool_calls: tuple[StreamedToolCall, ...] = (),
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost_usd: float = 0.001,
) -> ChatCompletionResult:
    return ChatCompletionResult(
        request_id="request-id",
        generation_id="generation-id",
        model="test/model",
        content=content,
        reasoning="",
        tool_calls=tool_calls,
        finish_reason=(
            "tool_calls" if tool_calls else "stop"
        ),
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=0,
            reasoning_tokens=0,
            cost_usd=cost_usd,
        ),
        request_started_ns=1_000_000_000,
        first_output_ns=1_050_000_000,
        last_output_ns=1_100_000_000,
        request_finished_ns=1_150_000_000,
        stream_event_timestamps_ns=(
            1_050_000_000,
            1_100_000_000,
        ),
        raw_event_count=2,
    )


def test_harbor_entrypoint_edits_workspace_and_reports_usage(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="edit-call",
                        name="replace_text",
                        arguments=(
                            '{"path":"app.py",'
                            '"old":"VALUE = 1",'
                            '"new":"VALUE = 2"}'
                        ),
                    ),
                ),
            ),
            make_completion(
                content="Updated the value.",
                input_tokens=20,
                output_tokens=8,
                cost_usd=0.002,
            ),
        ]
    )

    summary = run_harbor_task(
        client,
        tmp_path,
        "Change VALUE from 1 to 2.",
        model="test/model",
    )

    assert (tmp_path / "app.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert summary.status == "completed"
    assert summary.final_answer == "Updated the value."
    assert summary.model_call_count == 2
    assert summary.tool_call_count == 1
    assert summary.input_tokens == 30
    assert summary.output_tokens == 13
    assert summary.cost_usd == 0.003
    assert summary.first_ttft_ms == 50.0


def test_harbor_entrypoint_reports_missing_usage(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    completion = make_completion(content="No change needed.")
    completion = ChatCompletionResult(
        request_id=completion.request_id,
        generation_id=completion.generation_id,
        model=completion.model,
        content=completion.content,
        reasoning=completion.reasoning,
        tool_calls=completion.tool_calls,
        finish_reason=completion.finish_reason,
        usage=None,
        request_started_ns=completion.request_started_ns,
        first_output_ns=completion.first_output_ns,
        last_output_ns=completion.last_output_ns,
        request_finished_ns=completion.request_finished_ns,
        stream_event_timestamps_ns=(
            completion.stream_event_timestamps_ns
        ),
        raw_event_count=completion.raw_event_count,
    )

    summary = run_harbor_task(
        FakeChatClient([completion]),
        tmp_path,
        "Inspect the repository.",
        model="test/model",
    )

    assert summary.input_tokens is None
    assert summary.output_tokens is None
    assert summary.cost_usd is None

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from agent_harness import (
    OpenRouterClient,
    OpenRouterError,
)


def make_clock(
    *timestamps: int,
) -> tuple[Iterator[int], Any]:
    values = iter(timestamps)
    return values, lambda: next(values)


def test_streams_content_usage_and_timings() -> None:
    stream = (
        ": OPENROUTER PROCESSING\n\n"
        'data: {"id":"req-1","model":"test/model",'
        '"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"id":"req-1","model":"test/model",'
        '"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        'data: {"id":"req-1","model":"test/model",'
        '"choices":[{"delta":{"content":" world"},'
        '"finish_reason":"stop"}]}\n\n'
        'data: {"id":"req-1","model":"test/model",'
        '"choices":[{"delta":{"content":""},'
        '"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,'
        '"completion_tokens":2,"total_tokens":12,'
        '"cost":0.001}}\n\n'
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "X-Generation-Id": "gen-1",
            },
            text=stream,
        )
    )
    http_client = httpx.Client(transport=transport)

    _, clock = make_clock(
        1_000_000_000,
        1_100_000_000,
        1_250_000_000,
        1_400_000_000,
    )

    client = OpenRouterClient(
        "test-key",
        http_client=http_client,
        clock_ns=clock,
    )

    result = client.complete(
        [{"role": "user", "content": "Hello"}],
        model="test/model",
    )

    assert result.content == "Hello world"
    assert result.request_id == "req-1"
    assert result.generation_id == "gen-1"
    assert result.finish_reason == "stop"

    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 2
    assert result.usage.cost_usd == 0.001

    assert result.ttft_ms == 100
    assert result.total_latency_ms == 400
    assert result.generation_duration_ms == 150
    assert result.stream_event_intervals_ms == (150.0,)
    assert result.output_tokens_per_second == pytest.approx(
        13.333333,
    )


def test_assembles_streamed_tool_calls() -> None:
    stream = (
        'data: {"id":"req-tools","choices":[{"delta":'
        '{"tool_calls":[{"index":0,"id":"call-1",'
        '"function":{"name":"read_file",'
        '"arguments":"{\\"path\\":"}}]}}]}\n\n'
        'data: {"id":"req-tools","choices":[{"delta":'
        '{"tool_calls":[{"index":0,"function":'
        '{"arguments":"\\"app.py\\"}"}}]},'
        '"finish_reason":"tool_calls"}]}\n\n'
        'data: {"id":"req-tools","choices":[],'
        '"usage":{"prompt_tokens":20,'
        '"completion_tokens":5,"total_tokens":25}}\n\n'
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=stream,
        )
    )

    _, clock = make_clock(
        1_000_000_000,
        1_050_000_000,
        1_100_000_000,
        1_150_000_000,
    )

    client = OpenRouterClient(
        "test-key",
        http_client=httpx.Client(transport=transport),
        clock_ns=clock,
    )

    result = client.complete(
        [{"role": "user", "content": "Read app.py"}],
        model="test/model",
    )

    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.call_id == "call-1"
    assert tool_call.name == "read_file"
    assert tool_call.arguments == '{"path":"app.py"}'
    assert result.finish_reason == "tool_calls"


def test_builds_streaming_request_payload() -> None:
    captured_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(
            json.loads(request.content)
        )

        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"id":"req-2","choices":'
                '[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    transport = httpx.MockTransport(handler)
    _, clock = make_clock(
        1_000_000_000,
        1_100_000_000,
        1_200_000_000,
    )

    client = OpenRouterClient(
        "test-key",
        http_client=httpx.Client(transport=transport),
        clock_ns=clock,
    )

    client.complete(
        [{"role": "user", "content": "Test"}],
        model="test/model",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                    },
                },
            }
        ],
        temperature=0.0,
        seed=42,
        session_id="session-1",
    )

    assert captured_payload["stream"] is True
    assert captured_payload["stream_options"] == {
        "include_usage": True
    }
    assert captured_payload["tool_choice"] == "auto"
    assert captured_payload["temperature"] == 0.0
    assert captured_payload["seed"] == 42
    assert captured_payload["session_id"] == "session-1"


def test_raises_for_http_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                }
            },
        )
    )

    client = OpenRouterClient(
        "test-key",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(
        OpenRouterError,
        match="HTTP 401: Invalid API key",
    ):
        client.complete(
            [{"role": "user", "content": "Hello"}],
            model="test/model",
        )


def test_raises_for_midstream_error() -> None:
    stream = (
        'data: {"error":{"code":"server_error",'
        '"message":"Provider disconnected"},'
        '"choices":[{"delta":{"content":""},'
        '"finish_reason":"error"}]}\n\n'
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=stream,
        )
    )

    client = OpenRouterClient(
        "test-key",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(
        OpenRouterError,
        match="Provider disconnected",
    ):
        client.complete(
            [{"role": "user", "content": "Hello"}],
            model="test/model",
        )


def test_rejects_malformed_stream_json() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text="data: {not-json}\n\n",
        )
    )

    client = OpenRouterClient(
        "test-key",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(
        OpenRouterError,
        match="malformed JSON",
    ):
        client.complete(
            [{"role": "user", "content": "Hello"}],
            model="test/model",
        )

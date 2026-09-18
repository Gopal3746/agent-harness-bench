from __future__ import annotations

import json
import time
from collections.abc import (
    Callable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import httpx

_DEFAULT_ENDPOINT = (
    "https://openrouter.ai/api/v1/chat/completions"
)


class OpenRouterError(RuntimeError):
    """An OpenRouter request or stream failed."""


@dataclass(frozen=True, slots=True)
class StreamedToolCall:
    """A function call assembled from streamed argument fragments."""

    index: int
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token and cost information reported by OpenRouter."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    """Aggregated output and timing data for one model request."""

    request_id: str | None
    generation_id: str | None
    model: str | None
    content: str
    reasoning: str
    tool_calls: tuple[StreamedToolCall, ...]
    finish_reason: str | None
    usage: TokenUsage | None
    request_started_ns: int
    first_output_ns: int | None
    last_output_ns: int | None
    request_finished_ns: int
    stream_event_timestamps_ns: tuple[int, ...]
    raw_event_count: int

    @property
    def ttft_ms(self) -> float | None:
        if self.first_output_ns is None:
            return None

        return (
            self.first_output_ns - self.request_started_ns
        ) / 1_000_000

    @property
    def total_latency_ms(self) -> float:
        return (
            self.request_finished_ns - self.request_started_ns
        ) / 1_000_000

    @property
    def generation_duration_ms(self) -> float | None:
        if (
            self.first_output_ns is None
            or self.last_output_ns is None
        ):
            return None

        return (
            self.last_output_ns - self.first_output_ns
        ) / 1_000_000

    @property
    def stream_event_intervals_ms(self) -> tuple[float, ...]:
        timestamps = self.stream_event_timestamps_ns

        return tuple(
            (current - previous) / 1_000_000
            for previous, current in pairwise(timestamps)
        )

    @property
    def mean_stream_event_interval_ms(self) -> float | None:
        intervals = self.stream_event_intervals_ms

        if not intervals:
            return None

        return sum(intervals) / len(intervals)

    @property
    def output_tokens_per_second(self) -> float | None:
        if self.usage is None:
            return None

        duration_ms = self.generation_duration_ms
        if duration_ms is None or duration_ms <= 0:
            return None

        return self.usage.output_tokens / (
            duration_ms / 1_000
        )


@dataclass(slots=True)
class _ToolCallBuffer:
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class OpenRouterClient:
    """OpenAI-compatible streaming client with timing capture."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout_seconds: float = 180.0,
        http_client: httpx.Client | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        app_name: str | None = None,
        site_url: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.api_key = api_key
        self.endpoint = endpoint
        self.clock_ns = clock_ns
        self.app_name = app_name
        self.site_url = site_url

        self._owns_client = http_client is None
        self._http_client = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds)
        )

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
        """Stream and aggregate one chat-completion request."""

        if not messages:
            raise ValueError("messages must not be empty")

        if not model.strip():
            raise ValueError("model must not be blank")

        payload: dict[str, Any] = {
            "messages": list(messages),
            "model": model,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if seed is not None:
            payload["seed"] = seed

        if session_id is not None:
            payload["session_id"] = session_id

        return self._stream(payload)

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def _stream(
        self,
        payload: dict[str, Any],
    ) -> ChatCompletionResult:
        started_ns = self.clock_ns()

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        event_timestamps: list[int] = []
        tool_buffers: dict[int, _ToolCallBuffer] = {}

        request_id: str | None = None
        generation_id: str | None = None
        response_model: str | None = None
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        raw_event_count = 0

        with self._http_client.stream(
            "POST",
            self.endpoint,
            headers=self._headers(),
            json=payload,
        ) as response:
            generation_id = response.headers.get(
                "X-Generation-Id"
            )

            if response.status_code >= 400:
                response.read()
                raise self._http_error(response)

            for event_data in _iter_sse_data(response):
                if event_data == "[DONE]":
                    break

                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError as error:
                    raise OpenRouterError(
                        "received malformed JSON in SSE stream"
                    ) from error

                raw_event_count += 1

                if "error" in event:
                    raise OpenRouterError(
                        _extract_error_message(event["error"])
                    )

                request_id = event.get("id") or request_id
                response_model = (
                    event.get("model") or response_model
                )

                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = _parse_usage(event_usage)

                choices = event.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                choice_finish_reason = choice.get(
                    "finish_reason"
                )
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason

                delta = choice.get("delta") or {}
                meaningful_output = False

                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    meaningful_output = True

                reasoning = _extract_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    meaningful_output = True

                tool_deltas = delta.get("tool_calls") or []
                if tool_deltas:
                    meaningful_output = True
                    _apply_tool_call_deltas(
                        tool_buffers,
                        tool_deltas,
                    )

                if meaningful_output:
                    event_timestamps.append(self.clock_ns())

        finished_ns = self.clock_ns()

        tool_calls = tuple(
            StreamedToolCall(
                index=buffer.index,
                call_id=buffer.call_id,
                name=buffer.name,
                arguments=buffer.arguments,
            )
            for _, buffer in sorted(tool_buffers.items())
        )

        return ChatCompletionResult(
            request_id=request_id,
            generation_id=generation_id,
            model=response_model,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            request_started_ns=started_ns,
            first_output_ns=(
                event_timestamps[0]
                if event_timestamps
                else None
            ),
            last_output_ns=(
                event_timestamps[-1]
                if event_timestamps
                else None
            ),
            request_finished_ns=finished_ns,
            stream_event_timestamps_ns=tuple(
                event_timestamps
            ),
            raw_event_count=raw_event_count,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.app_name:
            headers["X-Title"] = self.app_name

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

        return headers

    @staticmethod
    def _http_error(
        response: httpx.Response,
    ) -> OpenRouterError:
        try:
            payload = response.json()
            message = _extract_error_message(
                payload.get("error", payload)
            )
        except (ValueError, AttributeError):
            message = response.text or "unknown HTTP error"

        return OpenRouterError(
            f"OpenRouter returned HTTP "
            f"{response.status_code}: {message}"
        )


def _iter_sse_data(
    response: httpx.Response,
) -> Iterator[str]:
    """Parse SSE framing while ignoring comments and other fields."""

    data_lines: list[str] = []

    for line in response.iter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator or field != "data":
            continue

        value = value.removeprefix(" ")

        data_lines.append(value)

    if data_lines:
        yield "\n".join(data_lines)


def _apply_tool_call_deltas(
    buffers: dict[int, _ToolCallBuffer],
    deltas: Sequence[dict[str, Any]],
) -> None:
    for delta in deltas:
        index = int(delta.get("index", 0))
        buffer = buffers.setdefault(
            index,
            _ToolCallBuffer(index=index),
        )

        call_id = delta.get("id")
        if isinstance(call_id, str) and call_id:
            buffer.call_id = call_id

        function = delta.get("function") or {}

        name = function.get("name")
        if isinstance(name, str) and name:
            buffer.name += name

        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            buffer.arguments += arguments


def _extract_reasoning(delta: dict[str, Any]) -> str:
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value

    return ""


def _parse_usage(usage: dict[str, Any]) -> TokenUsage:
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = (
        usage.get("completion_tokens_details") or {}
    )

    cost = usage.get("cost")

    return TokenUsage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(
            usage.get("completion_tokens") or 0
        ),
        total_tokens=int(usage.get("total_tokens") or 0),
        cached_tokens=int(
            prompt_details.get("cached_tokens") or 0
        ),
        reasoning_tokens=int(
            completion_details.get("reasoning_tokens") or 0
        ),
        cost_usd=(
            float(cost)
            if isinstance(cost, (int, float))
            else None
        ),
    )


def _extract_error_message(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message

        return json.dumps(
            error,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(error)

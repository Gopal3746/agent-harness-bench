from __future__ import annotations

import json
import time
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from agent_harness.llm import (
    ChatCompletionResult,
    OpenRouterError,
)
from agent_harness.models import TaskSpec
from agent_harness.tools import (
    ToolDispatcher,
    build_tool_schemas,
)

_SYSTEM_PROMPT = """\
You are a coding agent operating inside an isolated repository workspace.

Inspect the repository before editing it. Make the smallest correct change
needed to complete the task. Use the available tools for reading, searching,
editing, and running commands. Run the configured tests before claiming the
task is complete. If a tool fails, inspect the error and recover when possible.
Do not access files outside the assigned workspace.
"""


class ChatClient(Protocol):
    """Interface required by the single-agent loop."""

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
        """Return one streamed model completion."""


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One tool call and its serialized response."""

    step: int
    call_id: str
    name: str
    arguments: str
    response: str
    is_error: bool
    duration_ms: float


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Complete result and trace of one single-agent task run."""

    task_id: str
    status: AgentStatus
    final_answer: str
    error: str | None
    steps: int
    wall_time_ms: float
    messages: tuple[dict[str, Any], ...]
    model_calls: tuple[ChatCompletionResult, ...]
    tool_calls: tuple[ToolCallRecord, ...]

    @property
    def completed(self) -> bool:
        return self.status is AgentStatus.COMPLETED

    @property
    def total_model_calls(self) -> int:
        return len(self.model_calls)

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(
            call.usage.input_tokens
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            call.usage.output_tokens
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(
            call.usage.cost_usd or 0.0
            for call in self.model_calls
            if call.usage is not None
        )

    @property
    def total_model_latency_ms(self) -> float:
        return sum(
            call.total_latency_ms
            for call in self.model_calls
        )

    @property
    def total_tool_duration_ms(self) -> float:
        return sum(
            call.duration_ms
            for call in self.tool_calls
        )


class SingleAgent:
    """Runs one coding agent through a model/tool loop."""

    def __init__(
        self,
        client: ChatClient,
        dispatcher: ToolDispatcher,
        *,
        model: str,
        max_steps: int = 20,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = 42,
        session_id: str | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")

        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        if not 0 <= temperature <= 2:
            raise ValueError(
                "temperature must be between 0 and 2"
            )

        self.client = client
        self.dispatcher = dispatcher
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.session_id = session_id
        self.clock_ns = clock_ns
        self.tool_schemas = build_tool_schemas()

    def run(self, task: TaskSpec) -> AgentRunResult:
        """Run the agent until completion or its step limit."""

        started_ns = self.clock_ns()
        messages = self._initial_messages(task)

        model_calls: list[ChatCompletionResult] = []
        tool_records: list[ToolCallRecord] = []
        final_answer = ""

        for step in range(1, self.max_steps + 1):
            try:
                completion = self.client.complete(
                    messages,
                    model=self.model,
                    tools=self.tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    session_id=self.session_id,
                )
            except OpenRouterError as error:
                return self._result(
                    task=task,
                    status=AgentStatus.FAILED,
                    final_answer=final_answer,
                    error=str(error),
                    steps=step,
                    started_ns=started_ns,
                    messages=messages,
                    model_calls=model_calls,
                    tool_records=tool_records,
                )

            model_calls.append(completion)

            if completion.content:
                final_answer = completion.content

            if not completion.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.content,
                    }
                )

                return self._result(
                    task=task,
                    status=AgentStatus.COMPLETED,
                    final_answer=final_answer,
                    error=None,
                    steps=step,
                    started_ns=started_ns,
                    messages=messages,
                    model_calls=model_calls,
                    tool_records=tool_records,
                )

            resolved_calls = []

            for position, tool_call in enumerate(
                completion.tool_calls
            ):
                call_id = (
                    tool_call.call_id
                    or f"call-{step}-{position}"
                )

                resolved_calls.append(
                    (
                        call_id,
                        tool_call.name,
                        tool_call.arguments,
                    )
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": completion.content or None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                        for call_id, name, arguments
                        in resolved_calls
                    ],
                }
            )

            for call_id, name, arguments in resolved_calls:
                response = self.dispatcher.dispatch(
                    name,
                    arguments,
                )

                tool_records.append(
                    ToolCallRecord(
                        step=step,
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        response=response.content,
                        is_error=response.is_error,
                        duration_ms=response.duration_ms,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": response.content,
                    }
                )

        return self._result(
            task=task,
            status=AgentStatus.MAX_STEPS,
            final_answer=final_answer,
            error=(
                f"agent reached the maximum of "
                f"{self.max_steps} model steps"
            ),
            steps=self.max_steps,
            started_ns=started_ns,
            messages=messages,
            model_calls=model_calls,
            tool_records=tool_records,
        )

    @staticmethod
    def _initial_messages(
        task: TaskSpec,
    ) -> list[dict[str, Any]]:
        verification_command = json.dumps(
            list(task.test_command)
        )

        return [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Task ID: {task.task_id}\n\n"
                    f"{task.instruction}\n\n"
                    "Configured verification command: "
                    f"{verification_command}"
                ),
            },
        ]

    def _result(
        self,
        *,
        task: TaskSpec,
        status: AgentStatus,
        final_answer: str,
        error: str | None,
        steps: int,
        started_ns: int,
        messages: list[dict[str, Any]],
        model_calls: list[ChatCompletionResult],
        tool_records: list[ToolCallRecord],
    ) -> AgentRunResult:
        finished_ns = self.clock_ns()

        return AgentRunResult(
            task_id=task.task_id,
            status=status,
            final_answer=final_answer,
            error=error,
            steps=steps,
            wall_time_ms=(
                finished_ns - started_ns
            )
            / 1_000_000,
            messages=tuple(messages),
            model_calls=tuple(model_calls),
            tool_calls=tuple(tool_records),
        )

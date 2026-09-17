from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from pydantic import ValidationError

from agent_harness.models import TaskSpec
from agent_harness.tools.command import (
    CommandResult,
    CommandRunner,
)
from agent_harness.tools.repository import (
    RepositoryTools,
    ToolExecutionError,
)
from agent_harness.tools.schemas import (
    TOOL_ARGUMENT_MODELS,
    ToolArguments,
)


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """Serialized response returned to the model after a tool call."""

    name: str
    content: str
    is_error: bool
    duration_ms: float


class ToolDispatcher:
    """Validates and dispatches model-generated function calls."""

    def __init__(
        self,
        repository: RepositoryTools,
        commands: CommandRunner,
        task: TaskSpec,
    ) -> None:
        self.repository = repository
        self.commands = commands
        self.task = task

    def dispatch(
        self,
        name: str,
        arguments_json: str,
    ) -> ToolResponse:
        """Validate, execute, and serialize one tool call."""

        started_ns = time.perf_counter_ns()
        argument_model = TOOL_ARGUMENT_MODELS.get(name)

        if argument_model is None:
            return self._error_response(
                name,
                f"unknown tool: {name}",
                started_ns,
            )

        try:
            arguments = argument_model.model_validate_json(
                arguments_json or "{}"
            )
        except ValidationError as error:
            return self._error_response(
                name,
                f"invalid tool arguments: {error}",
                started_ns,
            )

        try:
            result = self._execute(name, arguments)
        except (
            FileNotFoundError,
            OSError,
            ToolExecutionError,
            ValueError,
        ) as error:
            return self._error_response(
                name,
                str(error),
                started_ns,
            )

        duration_ms = self._elapsed_ms(started_ns)

        return ToolResponse(
            name=name,
            content=json.dumps(
                self._to_jsonable(result),
                ensure_ascii=False,
                sort_keys=True,
            ),
            is_error=False,
            duration_ms=duration_ms,
        )

    def _execute(
        self,
        name: str,
        arguments: ToolArguments,
    ) -> Any:
        values = arguments.model_dump()

        if name == "list_files":
            return self.repository.list_files(**values)

        if name == "read_file":
            return self.repository.read_file(**values)

        if name == "search_repo":
            return self.repository.search_repo(**values)

        if name == "write_file":
            return self.repository.write_file(**values)

        if name == "replace_text":
            return self.repository.replace_text(**values)

        if name == "run_command":
            return self.commands.run(**values)

        if name == "run_tests":
            return self.commands.run_tests(self.task)

        raise ToolExecutionError(f"unknown tool: {name}")

    def _error_response(
        self,
        name: str,
        message: str,
        started_ns: int,
    ) -> ToolResponse:
        return ToolResponse(
            name=name,
            content=json.dumps(
                {"error": message},
                ensure_ascii=False,
                sort_keys=True,
            ),
            is_error=True,
            duration_ms=self._elapsed_ms(started_ns),
        )

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, CommandResult):
            serialized = asdict(value)
            serialized["succeeded"] = value.succeeded
            return self._to_jsonable(serialized)

        if is_dataclass(value) and not isinstance(value, type):
            return self._to_jsonable(asdict(value))

        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                self._to_jsonable(item)
                for item in value
            ]

        return value

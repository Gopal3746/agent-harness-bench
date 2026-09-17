from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)


class ExecutionStrategy(StrEnum):
    """Supported coding-agent orchestration strategies."""

    SINGLE = "single"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class TaskSpec(BaseModel):
    """Definition of one repository-level coding task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Stable lowercase identifier for the task.",
    )
    instruction: str
    source_dir: Path
    test_command: tuple[str, ...] = ("pytest", "-q")
    timeout_seconds: PositiveInt = 300
    tags: tuple[str, ...] = ()

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("instruction must not be blank")
        return cleaned

    @field_validator("test_command")
    @classmethod
    def validate_test_command(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("test_command must contain at least one argument")

        if any(not argument.strip() for argument in value):
            raise ValueError("test_command arguments must not be blank")

        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        cleaned = tuple(tag.strip().lower() for tag in value)

        if any(not tag for tag in cleaned):
            raise ValueError("tags must not be blank")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("tags must be unique")

        return cleaned


class BenchmarkConfig(BaseModel):
    """Configuration shared by a controlled benchmark experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    strategies: tuple[ExecutionStrategy, ...] = (
        ExecutionStrategy.SINGLE,
    )
    repetitions: PositiveInt = 5
    parallelism: PositiveInt = 1
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_steps: PositiveInt = 20
    seed: int = 42

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must not be blank")
        return cleaned

    @field_validator("strategies")
    @classmethod
    def validate_strategies(
        cls,
        value: tuple[ExecutionStrategy, ...],
    ) -> tuple[ExecutionStrategy, ...]:
        if not value:
            raise ValueError("at least one strategy is required")

        if len(value) != len(set(value)):
            raise ValueError("strategies must be unique")

        return value

    @model_validator(mode="after")
    def validate_parallelism(self) -> BenchmarkConfig:
        if (
            ExecutionStrategy.PARALLEL in self.strategies
            and self.parallelism < 2
        ):
            raise ValueError(
                "parallelism must be at least 2 when using "
                "the parallel strategy"
            )

        return self

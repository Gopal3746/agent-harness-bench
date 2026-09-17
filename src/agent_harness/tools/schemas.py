from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class ToolArguments(BaseModel):
    """Base validation rules for model-generated tool arguments."""

    model_config = ConfigDict(extra="forbid")


class ListFilesArguments(ToolArguments):
    path: str = "."
    max_results: int = Field(default=200, ge=1, le=500)


class ReadFileArguments(ToolArguments):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> ReadFileArguments:
        if (
            self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError(
                "end_line must be greater than or equal to start_line"
            )

        return self


class SearchRepoArguments(ToolArguments):
    query: str = Field(min_length=1)
    path: str = "."
    case_sensitive: bool = False
    max_results: int = Field(default=100, ge=1, le=500)


class WriteFileArguments(ToolArguments):
    path: str
    content: str
    overwrite: bool = True


class ReplaceTextArguments(ToolArguments):
    path: str
    old: str = Field(min_length=1)
    new: str
    expected_replacements: int = Field(default=1, ge=1)


class RunCommandArguments(ToolArguments):
    command: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
    )


class RunTestsArguments(ToolArguments):
    pass


_TOOL_SPECS: tuple[
    tuple[str, str, type[ToolArguments]],
    ...,
] = (
    (
        "list_files",
        "List files within the assigned repository workspace.",
        ListFilesArguments,
    ),
    (
        "read_file",
        "Read a bounded line range from a UTF-8 repository file.",
        ReadFileArguments,
    ),
    (
        "search_repo",
        "Search repository text files for a literal string.",
        SearchRepoArguments,
    ),
    (
        "write_file",
        "Create or atomically overwrite a UTF-8 repository file.",
        WriteFileArguments,
    ),
    (
        "replace_text",
        (
            "Replace an exact number of occurrences of text in a "
            "repository file."
        ),
        ReplaceTextArguments,
    ),
    (
        "run_command",
        (
            "Run an allowlisted command without shell expansion inside "
            "the repository workspace."
        ),
        RunCommandArguments,
    ),
    (
        "run_tests",
        "Run the coding task's configured verification command.",
        RunTestsArguments,
    ),
)

TOOL_ARGUMENT_MODELS: dict[str, type[ToolArguments]] = {
    name: argument_model
    for name, _, argument_model in _TOOL_SPECS
}


def build_tool_schemas() -> tuple[dict[str, Any], ...]:
    """Build OpenAI-compatible function-calling schemas."""

    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": argument_model.model_json_schema(),
            },
        }
        for name, description, argument_model in _TOOL_SPECS
    )

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agent_harness.models import TaskSpec
from agent_harness.tools.repository import ToolExecutionError

_DEFAULT_ALLOWED_COMMANDS = frozenset(
    {
        "pytest",
        "python",
        "python3",
        "ruff",
    }
)

_RUNTIME_DIRECTORY = ".agent-runtime"
_TRUNCATION_MARKER = "\n... output truncated ...\n"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of one command execution."""

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    output_truncated: bool

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0


class CommandRunner:
    """Runs allowlisted commands inside a repository workspace."""

    def __init__(
        self,
        root: Path,
        *,
        allowed_commands: frozenset[str] | None = None,
        default_timeout_seconds: float = 120.0,
        max_timeout_seconds: float = 600.0,
        max_output_chars: int = 20_000,
    ) -> None:
        resolved_root = root.expanduser().resolve()

        if not resolved_root.is_dir():
            raise FileNotFoundError(
                f"repository root does not exist: {resolved_root}"
            )

        if default_timeout_seconds <= 0:
            raise ValueError(
                "default_timeout_seconds must be positive"
            )

        if max_timeout_seconds < default_timeout_seconds:
            raise ValueError(
                "max_timeout_seconds must be greater than or equal "
                "to default_timeout_seconds"
            )

        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")

        self.root = resolved_root
        self.allowed_commands = (
            allowed_commands or _DEFAULT_ALLOWED_COMMANDS
        )
        self.default_timeout_seconds = default_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self.max_output_chars = max_output_chars

        runtime_directory = self.root / _RUNTIME_DIRECTORY
        self._home_directory = runtime_directory / "home"
        self._temp_directory = runtime_directory / "tmp"

        self._home_directory.mkdir(parents=True, exist_ok=True)
        self._temp_directory.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path = ".",
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Execute a command without invoking a shell."""

        normalized_command = self._validate_command(command)
        working_directory = self._resolve_working_directory(cwd)
        resolved_timeout = self._resolve_timeout(timeout_seconds)

        started_ns = time.perf_counter_ns()

        process = subprocess.Popen(
            normalized_command,
            cwd=working_directory,
            env=self._build_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        timed_out = False

        try:
            stdout, stderr = process.communicate(
                timeout=resolved_timeout
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process_group(process)
            stdout, stderr = process.communicate()

        duration_ms = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000

        truncated_stdout, stdout_truncated = self._truncate(stdout)
        truncated_stderr, stderr_truncated = self._truncate(stderr)

        return CommandResult(
            command=normalized_command,
            exit_code=None if timed_out else process.returncode,
            stdout=truncated_stdout,
            stderr=truncated_stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
            output_truncated=(
                stdout_truncated or stderr_truncated
            ),
        )

    def run_tests(self, task: TaskSpec) -> CommandResult:
        """Execute a task's configured verification command."""

        return self.run(
            task.test_command,
            timeout_seconds=float(task.timeout_seconds),
        )

    def _validate_command(
        self,
        command: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(command, (str, bytes)):
            raise ToolExecutionError(
                "command must be a sequence of arguments"
            )

        normalized = tuple(command)

        if not normalized:
            raise ToolExecutionError("command must not be empty")

        if any(
            not isinstance(argument, str) or not argument
            for argument in normalized
        ):
            raise ToolExecutionError(
                "command arguments must be non-empty strings"
            )

        if any("\0" in argument for argument in normalized):
            raise ToolExecutionError(
                "command arguments must not contain null bytes"
            )

        executable = normalized[0]

        if Path(executable).name != executable:
            raise ToolExecutionError(
                "executable paths are not allowed"
            )

        if executable not in self.allowed_commands:
            raise ToolExecutionError(
                f"command is not allowed: {executable}"
            )

        return normalized

    def _resolve_working_directory(
        self,
        cwd: str | Path,
    ) -> Path:
        supplied_path = Path(cwd)

        if supplied_path.is_absolute():
            raise ToolExecutionError(
                "absolute working directories are not allowed"
            )

        if _RUNTIME_DIRECTORY in supplied_path.parts:
            raise ToolExecutionError(
                "runtime directory cannot be used as a working directory"
            )

        resolved = (self.root / supplied_path).resolve()

        if not resolved.is_relative_to(self.root):
            raise ToolExecutionError(
                "working directory escapes the repository workspace"
            )

        if not resolved.is_dir():
            raise ToolExecutionError(
                f"working directory does not exist: {cwd}"
            )

        return resolved

    def _resolve_timeout(
        self,
        timeout_seconds: float | None,
    ) -> float:
        resolved = (
            self.default_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )

        if resolved <= 0:
            raise ToolExecutionError(
                "timeout_seconds must be positive"
            )

        if resolved > self.max_timeout_seconds:
            raise ToolExecutionError(
                f"timeout_seconds exceeds maximum of "
                f"{self.max_timeout_seconds}"
            )

        return resolved

    def _build_environment(self) -> dict[str, str]:
        environment = {
            "HOME": str(self._home_directory),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(self._temp_directory),
        }

        if "LC_ALL" in os.environ:
            environment["LC_ALL"] = os.environ["LC_ALL"]

        if "VIRTUAL_ENV" in os.environ:
            environment["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

        return environment

    def _truncate(self, output: str) -> tuple[str, bool]:
        if len(output) <= self.max_output_chars:
            return output, False

        marker_length = len(_TRUNCATION_MARKER)
        remaining = max(
            self.max_output_chars - marker_length,
            0,
        )
        beginning_length = remaining // 2
        ending_length = remaining - beginning_length

        beginning = output[:beginning_length]
        ending = (
            output[-ending_length:]
            if ending_length
            else ""
        )

        return (
            beginning + _TRUNCATION_MARKER + ending,
            True,
        )

    @staticmethod
    def _terminate_process_group(
        process: subprocess.Popen[str],
    ) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

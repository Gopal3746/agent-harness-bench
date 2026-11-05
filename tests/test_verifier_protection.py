from __future__ import annotations

import json
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_harness import (
    BenchmarkConfig,
    ChatCompletionResult,
    RepositoryTools,
    SingleTaskRunner,
    StreamedToolCall,
    TaskSpec,
    ToolExecutionError,
    WorkspaceManager,
)
from agent_harness.workspace import VerifierSnapshot


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
        del messages, model, tools, temperature, max_tokens, seed
        del session_id
        return self.responses.pop(0)


def make_completion(
    *,
    content: str = "",
    tool_calls: tuple[StreamedToolCall, ...] = (),
) -> ChatCompletionResult:
    return ChatCompletionResult(
        request_id="request-id",
        generation_id="generation-id",
        model="test/model",
        content=content,
        reasoning="",
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=None,
        request_started_ns=1_000_000_000,
        first_output_ns=1_050_000_000,
        last_output_ns=1_100_000_000,
        request_finished_ns=1_150_000_000,
        stream_event_timestamps_ns=(1_050_000_000,),
        raw_event_count=1,
    )


def test_task_spec_protects_tests_by_default() -> None:
    task = TaskSpec(
        task_id="protected-tests",
        instruction="Fix the implementation.",
        source_dir=Path("repository"),
    )

    assert task.protected_paths == ("tests",)


@pytest.mark.parametrize(
    "protected_path",
    ["", ".", "..", "../tests", "nested/tests", "/tests"],
)
def test_task_spec_rejects_unsafe_protected_paths(
    protected_path: str,
) -> None:
    with pytest.raises(ValidationError, match="protected_paths"):
        TaskSpec(
            task_id="protected-tests",
            instruction="Fix the implementation.",
            source_dir=Path("repository"),
            protected_paths=(protected_path,),
        )


def test_verifier_snapshot_restores_exact_protected_tree(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    nested = tests / "nested"
    nested.mkdir(parents=True)
    original = nested / "test_app.py"
    original.write_bytes(b"assert VALUE == 2\n")
    original.chmod(0o640)
    nested.chmod(0o750)

    snapshot = VerifierSnapshot.capture(tmp_path, ("tests",))

    original.write_bytes(b"assert True\n")
    original.chmod(0o777)
    (tests / "added.py").write_text("assert True\n")

    snapshot.restore()

    assert original.read_bytes() == b"assert VALUE == 2\n"
    assert stat.S_IMODE(original.stat().st_mode) == 0o640
    assert stat.S_IMODE(nested.stat().st_mode) == 0o750
    assert not (tests / "added.py").exists()


def test_verifier_snapshot_replaces_symlink_attack(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("assert False\n")
    snapshot = VerifierSnapshot.capture(tmp_path, ("tests",))

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_app.py").write_text("assert True\n")
    for child in tests.iterdir():
        child.unlink()
    tests.rmdir()
    tests.symlink_to(outside, target_is_directory=True)

    snapshot.restore()

    assert not tests.is_symlink()
    assert (tests / "test_app.py").read_text() == "assert False\n"
    assert (outside / "test_app.py").read_text() == "assert True\n"


def test_snapshot_removes_new_protected_path(
    tmp_path: Path,
) -> None:
    snapshot = VerifierSnapshot.capture(tmp_path, ("tests",))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_fake.py").write_text("assert True\n")

    snapshot.restore()

    assert not tests.exists()


def test_repository_file_tools_reject_protected_paths(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_app.py"
    test_file.write_text("assert False\n")
    tools = RepositoryTools(
        tmp_path,
        protected_paths=("tests",),
    )

    with pytest.raises(ToolExecutionError, match="protected"):
        tools.write_file("tests/test_fake.py", "assert True\n")

    with pytest.raises(ToolExecutionError, match="protected"):
        tools.replace_text(
            "tests/test_app.py",
            "False",
            "True",
        )

    assert test_file.read_text() == "assert False\n"


def test_runner_restores_tests_changed_by_command(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    tests = source / "tests"
    tests.mkdir(parents=True)
    (source / "app.py").write_text("VALUE = 1\n")
    original_test = (
        "from app import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 2\n"
    )
    (tests / "test_app.py").write_text(original_test)

    command = (
        "from pathlib import Path; "
        "Path('tests/test_app.py').write_text('assert True\\n')"
    )
    client = FakeChatClient(
        [
            make_completion(
                tool_calls=(
                    StreamedToolCall(
                        index=0,
                        call_id="cheat",
                        name="run_command",
                        arguments=json.dumps(
                            {
                                "command": ["python", "-c", command],
                            }
                        ),
                    ),
                )
            ),
            make_completion(content="Done."),
        ]
    )
    task = TaskSpec(
        task_id="protected-verifier",
        instruction="Change VALUE to 2.",
        source_dir=source,
        test_command=("python", "-m", "pytest", "-q"),
        timeout_seconds=30,
    )

    result = SingleTaskRunner(
        client,
        WorkspaceManager(
            tmp_path / "workspaces",
            retain_workspaces=True,
        ),
        BenchmarkConfig(model="test/model", repetitions=1),
    ).run(task, run_id="protected-run")

    assert result.agent.completed is True
    assert result.verification.succeeded is False
    assert result.passed is False
    assert (
        result.workspace.path / "tests" / "test_app.py"
    ).read_text() == original_test

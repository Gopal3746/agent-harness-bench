from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class AgentHarnessHarborAgent(BaseAgent):
    """Harbor adapter for the instrumented single-agent harness."""

    capabilities = AgentCapabilities()

    _REMOTE_ROOT = "/opt/agent-harness-bench"

    def __init__(
        self,
        *args: Any,
        max_steps: int = 20,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.max_steps = int(max_steps)
        self.temperature = float(temperature)
        self.seed = int(seed)

        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "temperature must be between 0 and 2"
            )

        kwargs["skip_model_parsing"] = True
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "agent-harness-bench"

    def version(self) -> str:
        return "0.1.0"

    async def setup(
        self,
        environment: BaseEnvironment,
    ) -> None:
        package_dir = Path(__file__).resolve().parents[1]
        remote_package_dir = (
            f"{self._REMOTE_ROOT}/agent_harness"
        )

        create_result = await environment.exec(
            f"mkdir -p {shlex.quote(self._REMOTE_ROOT)}",
            timeout_sec=30,
            user="root",
        )

        if create_result.return_code != 0:
            raise RuntimeError(
                "failed to create Harbor agent directory: "
                f"{create_result.stderr or create_result.stdout}"
            )

        await environment.upload_dir(
            package_dir,
            remote_package_dir,
        )

        install_command = shlex.join(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "httpx>=0.28,<1",
                "matplotlib>=3.9,<4",
                "pydantic>=2.11,<3",
            ]
        )
        install_result = await environment.exec(
            install_command,
            timeout_sec=360,
            user="root",
        )

        if install_result.return_code != 0:
            raise RuntimeError(
                "failed to install Agent Harness dependencies: "
                f"{install_result.stderr or install_result.stdout}"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError(
                "a model name is required for Agent Harness Bench"
            )

        if not self.extra_env.get(
            "OPENROUTER_API_KEY",
            "",
        ).strip():
            raise ValueError(
                "OPENROUTER_API_KEY must be passed with --ae"
            )

        working_directory = await self._working_directory(
            environment
        )
        command = shlex.join(
            [
                "python",
                "-m",
                "agent_harness.integrations.harbor_entrypoint",
                "--instruction",
                instruction,
                "--model",
                self.model_name,
                "--root",
                working_directory,
                "--max-steps",
                str(self.max_steps),
                "--temperature",
                str(self.temperature),
                "--seed",
                str(self.seed),
                "--session-id",
                self.session_id or "harbor-task",
            ]
        )
        runtime_environment = {
            **self.extra_env,
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "PYTHONPATH": self._REMOTE_ROOT,
        }
        result = await environment.exec(
            command,
            cwd=working_directory,
            env=runtime_environment,
            timeout_sec=None,
        )

        self._write_logs(result.stdout, result.stderr)
        summary = self._parse_summary(result.stdout)
        context.n_input_tokens = summary.get("input_tokens")
        context.n_output_tokens = summary.get("output_tokens")
        context.cost_usd = summary.get("cost_usd")
        context.metadata = {
            **(context.metadata or {}),
            "agent_harness": summary,
        }

        if result.return_code != 0:
            raise RuntimeError(
                "Agent Harness Bench failed inside Harbor: "
                f"{summary.get('error') or result.stderr}"
            )

    async def _working_directory(
        self,
        environment: BaseEnvironment,
    ) -> str:
        result = await environment.exec(
            "pwd",
            timeout_sec=30,
        )

        if result.return_code != 0:
            raise RuntimeError(
                "failed to determine Harbor working directory"
            )

        directory = result.stdout.strip()

        if not directory:
            raise RuntimeError(
                "Harbor working directory was empty"
            )

        return directory

    def _write_logs(
        self,
        stdout: str,
        stderr: str,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "agent-harness.stdout.log").write_text(
            stdout,
            encoding="utf-8",
        )
        (self.logs_dir / "agent-harness.stderr.log").write_text(
            stderr,
            encoding="utf-8",
        )

    @staticmethod
    def _parse_summary(stdout: str) -> dict[str, Any]:
        for line in reversed(stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and "status" in payload:
                return payload

        raise RuntimeError(
            "Agent Harness Bench did not return a JSON summary"
        )

import pytest

from agent_harness.cli import main


def test_cli_runs_without_arguments() -> None:
    assert main([]) == 0


def test_cli_reports_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "agent-harness 0.1.0"

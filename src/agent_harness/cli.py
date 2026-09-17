import argparse
from collections.abc import Sequence

PROGRAM_DESCRIPTION = (
    "Run and evaluate instrumented coding-agent workloads."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version="agent-harness 0.1.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0

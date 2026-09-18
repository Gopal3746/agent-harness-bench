# Agent Harness Bench

An instrumented coding-agent harness for studying how orchestration strategies
affect correctness, latency, inference workload characteristics, and cost.

The project will compare single-agent, sequential-subagent, and
parallel-subagent execution on reproducible repository-level coding tasks.

## Planned measurements

- Task success and test pass rate
- Time to first streamed output
- End-to-end model latency
- Inter-stream-event latency
- Output-token throughput
- Task wall-clock time
- Input and output tokens
- Model calls and tool calls
- Estimated cost per task
- Parallel speedup

## Benchmark tasks

Task fixtures live under `benchmarks/tasks`. Each task contains an immutable
failing repository, an instruction, and a test command used for independent
verification. The harness copies the repository into a fresh workspace before
every run, ensuring that repeated experiments begin from the same state.

The first task, `input-validation`, requires coordinated changes across three
independent modules and is intentionally suitable for later sequential-versus-
parallel orchestration experiments.

## Development setup

Requires Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest
ruff check .
agent-harness --help
```

Benchmark results will be added only after controlled experiments have been
completed.

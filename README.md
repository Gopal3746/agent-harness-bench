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

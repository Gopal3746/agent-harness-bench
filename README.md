# Agent Harness Bench

[![CI](https://github.com/Gopal3746/agent-harness-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Gopal3746/agent-harness-bench/actions/workflows/ci.yml)

An instrumented coding-agent harness for studying how orchestration strategies
affect correctness, latency, token usage, tool activity, and inference cost.

Agent Harness Bench runs repository-level software-engineering tasks using
single-agent, sequential-subagent, and parallel-subagent architectures. It
executes each task in an isolated workspace, verifies the result with
task-specific tests, records structured telemetry, and generates comparative
analysis artifacts.

## Research questions

The project is designed to answer three questions:

1. When does task decomposition improve coding-agent performance?
2. What inference and coordination overhead does each strategy introduce?
3. Does parallel execution reduce end-to-end latency enough to justify its
   additional token usage and cost?

## Pilot result

A controlled 18-run pilot compared three strategies across six coding tasks
using `minimax/minimax-m3.1-flash`.

![Pilot strategy dashboard](experiments/pilot/strategy-dashboard.png)

| Strategy | Strict pass rate | Verification rate | Mean wall time | Mean model calls | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 66.7% | 100.0% | 21.27 s | 7.00 | $0.00247 |
| Sequential | 0.0% | 33.3% | 56.16 s | 12.67 | $0.00760 |
| Parallel | 33.3% | 50.0% | 56.10 s | 15.83 | $0.00822 |

The full pilot cost `$0.109798`.

A strict pass requires both successful independent verification and clean agent
termination. Eleven of 18 runs passed repository verification, while six
passed the strict end-to-end metric. Five runs produced correct changes but
reached the configured eight-step limit before terminating cleanly.

These results are exploratory. Each task and strategy was run once, so they
should not be interpreted as statistically significant.

## Key findings

### Single-agent execution performed best on small tasks

The single-agent strategy passed independent verification on all six tasks. It
was also the fastest and least expensive strategy.

For the small repositories in this suite, one agent had enough context to
complete the work without decomposition or merge overhead.

### Parallelism did not guarantee lower latency

Parallel execution increased mean wall time by `163.7%` relative to the
single-agent baseline. It also increased:

- mean input tokens by `113.1%`;
- mean output tokens by `267.9%`;
- cost per run by `232.4%`.

Parallel execution had the highest output-token throughput, but additional
planning, model calls, tool calls, and coordination eliminated the throughput
benefit at the task level.

### TTFT was not the main source of overhead

Mean time to first token remained relatively stable:

| Strategy | Mean TTFT |
| --- | ---: |
| Single | 757.68 ms |
| Sequential | 741.84 ms |
| Parallel | 774.10 ms |

The higher end-to-end latency came primarily from the number of inference and
tool-use steps rather than slower individual requests.

### Correctness and termination are different outcomes

Twelve runs reached the maximum-step condition. Five of those runs had already
produced code that passed independent verification.

This distinction motivates reporting four separate outcomes:

- independent verification success;
- clean agent completion;
- maximum-step termination;
- strict end-to-end success.

The complete methodology and results are available in
[`experiments/pilot`](experiments/pilot/README.md).

## Architecture

```mermaid
flowchart TD
    A["CLI and task catalog"] --> B["Isolated task workspace"]
    B --> C{"Execution strategy"}
    C --> D["Single agent"]
    C --> E["Sequential subagents"]
    C --> F["Parallel subagents"]
    D --> G["Model client and repository tools"]
    E --> G
    F --> G
    G --> H["Independent verification and telemetry"]
    H --> I["JSONL, CSV, and dashboard"]
```

## Execution strategies

### Single agent

One agent receives the complete task, inspects the repository, edits files, and
runs commands until it completes or reaches the configured step limit.

This is the lowest-overhead strategy and serves as the experimental baseline.

### Sequential subagents

A planner decomposes the task into dependency-aware subtasks. Workers execute
those subtasks in topological order so later workers can observe earlier
changes.

This strategy tests whether focused contexts improve correctness enough to
offset additional inference calls.

### Parallel subagents

A planner identifies independent subtasks, which run concurrently in cloned
workspaces. The harness calculates file-level change sets and merges compatible
changes into the final workspace.

Conflicting changes are rejected rather than silently overwritten.

## Coding-agent tools

Agents interact with repositories through structured tools:

- list repository files;
- read bounded file ranges;
- search repository text;
- write files;
- replace exact text;
- run task-approved commands;
- execute independent tests.

Repository paths are resolved against the workspace root. Absolute paths,
parent-directory traversal, and symlink escapes are rejected.

## Isolated execution and verification

Every task is provisioned into a fresh workspace. The source repository remains
unchanged, and temporary workspaces are removed unless retention is explicitly
enabled.

After the agent stops, the harness runs the task's verification command
independently. Agent claims are never treated as proof of correctness.

A run is considered a strict pass only when:

1. the agent terminates successfully; and
2. the independent verifier succeeds.

## Telemetry

Each run produces a structured JSONL record containing:

- task, model, strategy, and run identifiers;
- agent status and termination error;
- end-to-end wall time;
- model-call and tool-call counts;
- tool errors and tool execution duration;
- input, output, cached, and reasoning tokens;
- reported inference cost;
- time to first token;
- total model latency;
- output-token throughput;
- stream-event intervals;
- independent verification results.

Stream-event interval is recorded as an ITL-style proxy. It is not guaranteed
to represent exact token-level inter-token latency because providers may stream
multiple tokens in one event.

## Repository structure

```text
agent-harness-bench/
├── .github/
│   └── workflows/
│       └── ci.yml
├── benchmarks/
│   └── tasks/
│       ├── audit-events/
│       ├── config-merge/
│       ├── input-validation/
│       ├── pagination-client/
│       ├── service-retries/
│       └── ttl-cache/
├── experiments/
│   └── pilot/
│       ├── README.md
│       ├── runs.jsonl
│       ├── summary.csv
│       ├── task-summary.csv
│       ├── comparisons.csv
│       ├── strategy-dashboard.png
│       └── reproduce.sh
├── harbor/
│   ├── README.md
│   └── tasks/
│       └── value-update/
├── src/
│   └── agent_harness/
│       ├── agent/
│       ├── integrations/
│       ├── telemetry/
│       ├── tools/
│       ├── analysis.py
│       ├── benchmark.py
│       ├── cli.py
│       ├── models.py
│       ├── plotting.py
│       ├── runner.py
│       ├── safety.py
│       ├── suite.py
│       ├── task_catalog.py
│       └── workspace.py
└── tests/
```

## Requirements

- Python 3.12 or newer
- An OpenRouter API key for live model experiments
- Docker Desktop for Harbor container evaluation

Unit tests, CLI dry runs, task discovery, analysis, and Harbor schema validation
do not require paid model calls.

## Installation

```bash
git clone https://github.com/Gopal3746/agent-harness-bench.git
cd agent-harness-bench

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
agent-harness --version
agent-harness --help

ruff check .
python -m pytest
```

## Discover benchmark tasks

```bash
agent-harness tasks benchmarks/tasks
```

Each benchmark task contains:

- a natural-language instruction;
- a source repository;
- a reproducible failing baseline;
- an independent verification command;
- task tags describing its structure.

## Plan an experiment without spending money

The `--dry-run` option validates the suite and reports the planned number of
runs without calling a model:

```bash
agent-harness benchmark-suite \
  benchmarks/tasks \
  --model minimax/minimax-m3.1-flash \
  --strategy single \
  --strategy sequential \
  --strategy parallel \
  --repetitions 1 \
  --parallelism 2 \
  --max-steps 8 \
  --max-total-runs 18 \
  --dry-run
```

## Run one task

Set the API key in the environment. Never commit it to the repository.

```bash
export OPENROUTER_API_KEY="your-key"
```

Run a task with the single-agent strategy:

```bash
agent-harness run \
  benchmarks/tasks/input-validation/task.json \
  --model minimax/minimax-m3.1-flash \
  --strategy single \
  --max-steps 8 \
  --output results/single-run.jsonl \
  --workspace-root workspaces
```

Select another strategy with:

```text
--strategy sequential
--strategy parallel --parallelism 2
```

## Run a controlled suite

```bash
agent-harness benchmark-suite \
  benchmarks/tasks \
  --model minimax/minimax-m3.1-flash \
  --strategy single \
  --strategy sequential \
  --strategy parallel \
  --repetitions 1 \
  --parallelism 2 \
  --max-steps 8 \
  --temperature 0 \
  --seed 42 \
  --suite-id example-suite \
  --max-total-runs 18 \
  --output results/suite-runs.jsonl \
  --workspace-root workspaces
```

The harness refuses experiments that exceed `--max-total-runs`. It also refuses
to append to an existing non-empty output file unless overwrite behavior is
explicitly requested.

## Analyze results

```bash
agent-harness analyze \
  results/suite-runs.jsonl \
  --output results/summary.csv \
  --task-output results/task-summary.csv \
  --comparison-output results/comparisons.csv \
  --baseline single \
  --plot-output results/strategy-dashboard.png
```

The analysis reports:

- pass rate;
- mean and median wall time;
- TTFT;
- output-token throughput;
- model and tool calls;
- input and output tokens;
- total cost;
- baseline-relative speedup and percentage changes;
- per-task strategy results.

## Reproduce the pilot

The reproduction script refuses to overwrite an existing result file:

```bash
read -s "OPENROUTER_API_KEY?OpenRouter API key: "
export OPENROUTER_API_KEY
printf '\n'

OUTPUT_DIR=experiments/reproduction \
  ./experiments/pilot/reproduce.sh

unset OPENROUTER_API_KEY
```

Because this command performs live inference, reproduced values may differ due
to provider routing, model updates, and network conditions.

## Harbor integration

Install the optional Harbor dependency:

```bash
python -m pip install -e ".[dev,harbor]"
```

Validate the bundled Harbor task with its reference solution:

```bash
harbor run \
  -p harbor/tasks/value-update \
  --agent oracle
```

This reference run does not call a model. A successful result has a reward of
`1.000`.

Run Agent Harness Bench as a Harbor custom agent:

```bash
harbor run \
  -p harbor/tasks/value-update \
  --agent \
  agent_harness.integrations.harbor_agent:AgentHarnessHarborAgent \
  --model minimax/minimax-m3.1-flash \
  --ae OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
```

The adapter uploads the harness into Harbor's isolated task environment,
executes the coding-agent loop, and copies token, cost, latency, tool-use, and
status information into Harbor's agent context. Harbor then runs its independent
verifier.

See [`harbor/README.md`](harbor/README.md) for details.

## Continuous integration

GitHub Actions runs the following checks on pushes and pull requests:

- Python 3.12 installation;
- editable installation with development and Harbor dependencies;
- Ruff linting;
- the complete pytest suite;
- Harbor task-schema validation;
- custom Harbor agent import validation;
- API-free CLI and benchmark dry-run smoke tests.

The workflow does not require an OpenRouter secret.

## Experiment safety

The harness includes safeguards for paid experiments:

- explicit total-run limits;
- dry-run planning;
- refusal to overwrite non-empty outputs;
- isolated task workspaces;
- fixed seeds and temperatures;
- structured cost telemetry;
- API keys read only from environment variables.

Users should still monitor their provider balance during larger experiments.

## Limitations

The current implementation has several known limitations:

- The pilot has one repetition per task and strategy.
- Only one model was evaluated.
- Subagent strategies use the same per-agent step limit as the baseline.
- Parallel merges operate at the file-change level.
- The task suite contains small Python repositories.
- Stream events are only an approximation of token-level ITL.
- Model behavior may change as providers update weights or routing.
- The harness currently uses local process isolation for its native task suite;
  Harbor provides container isolation for framework-based evaluation.

## Next research steps

Future experiments should investigate:

1. multiple repetitions with confidence intervals;
2. separate planner and worker step budgets;
3. decomposition-confidence thresholds;
4. adaptive strategy selection;
5. parallelism levels of 1, 2, 4, and 8;
6. larger repository tasks;
7. multiple models and inference providers;
8. conflict-aware semantic merge strategies;
9. token-level streaming telemetry where providers expose it.

The pilot suggests that orchestration should be selected based on task
structure rather than enabled by default. For small tasks, coordination
overhead can dominate the potential benefit of parallel inference.

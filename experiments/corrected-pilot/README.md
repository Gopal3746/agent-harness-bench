# Corrected Orchestration Pilot

This directory contains a controlled rerun of the six-task orchestration
pilot after correcting harness behaviors identified during review of the
original experiment.

The rerun is descriptive rather than statistically conclusive: it contains
one run per task and strategy. Its purpose is to verify that the harness
corrections materially change the observed outcomes before spending more on
repeated trials.

## Configuration

| Setting | Value |
| --- | --- |
| Model | `minimax/minimax-m3.1-flash` |
| Tasks | 6 |
| Strategies | single, sequential, parallel |
| Repetitions | 1 |
| Total runs | 18 |
| Parallelism | 2 |
| Maximum steps | 9 per agent |
| Temperature | 0 |
| Seed | 42 |
| Suite ID | `corrected-pilot-2026-11-10` |

Nine steps preserve the original allowance of eight working turns while
reserving the ninth turn for a tool-free final answer.

## Corrections evaluated

The corrected harness:

- restores protected verifier assets before independent verification;
- gives each worker the original task, complete plan, and explicit ownership;
- asks workers to verify only their assigned subtask;
- reserves the final model turn for completion instead of another tool call;
- merges changes from successful parallel workers when a sibling fails;
- supports parallel execution with one worker;
- records bounded, redacted tool arguments and responses;
- records planning and merge phase durations; and
- preloads bounded planner-selected file context for workers.

## Results

| Strategy | Passed | Pass rate | Mean wall time | Mean model calls | Mean input tokens | Mean output tokens | Total cost | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 6/6 | 100.0% | 22.66 s | 6.83 | 21,023 | 2,856 | $0.01619 | $0.00270 |
| Sequential | 4/6 | 66.7% | 173.71 s | 33.67 | 138,429 | 22,679 | $0.12063 | $0.02010 |
| Parallel | 4/6 | 66.7% | 131.02 s | 21.50 | 90,117 | 25,252 | $0.11892 | $0.01982 |

Overall, 14 of 18 runs passed (77.8%) at a total reported cost of
$0.25574.

### Change from the original pilot

| Strategy | Original | Corrected | Change |
| --- | ---: | ---: | ---: |
| Single | 4/6 (66.7%) | 6/6 (100.0%) | +33.3 pp |
| Sequential | 0/6 (0.0%) | 4/6 (66.7%) | +66.7 pp |
| Parallel | 2/6 (33.3%) | 4/6 (66.7%) | +33.3 pp |
| Overall | 6/18 (33.3%) | 14/18 (77.8%) | +44.4 pp |

The original and corrected runs used the same model, tasks, temperature,
seed, and strategy set. The corrected run used nine steps rather than eight
to retain eight working turns plus the new required completion turn.

## Per-task outcomes

| Task | Single | Sequential | Parallel |
| --- | ---: | ---: | ---: |
| audit-events | Pass | Pass | Pass |
| config-merge | Pass | Pass | Fail |
| input-validation | Pass | Pass | Pass |
| pagination-client | Pass | Pass | Pass |
| service-retries | Pass | Fail | Pass |
| ttl-cache-expiration | Pass | Fail | Fail |

## Phase and overhead findings

- Mean planner time was 11.87 seconds for sequential and 10.16 seconds for
  parallel runs, about 6.8% and 7.8% of their respective wall times.
- Mean parallel merge time was only 1.54 milliseconds. File merging was not
  the source of the remaining parallel latency.
- Parallel matched sequential pass rate and was 24.6% faster on mean wall
  time, while costing 1.4% less.
- Compared with single-agent execution, sequential and parallel used far more
  model calls and tokens. Their mean cost per run was approximately 7.45x and
  7.35x the single-agent cost.

## Remaining failures

Four runs did not count as passes:

- `config-merge / parallel` completed, but independent verification failed.
- `service-retries / sequential` reached the nine-step limit in the
  `retry-tests` subtask. Independent verification nevertheless exited 0.
- `ttl-cache-expiration / sequential` reached the limit in the
  `size_expiration` subtask. Independent verification nevertheless exited 0.
- `ttl-cache-expiration / parallel` reached the limit in the
  `ttl-validation` subtask, and independent verification failed.

The two max-step runs whose work passed independent verification show that
completion status remains stricter than repository correctness. A future
experiment should report both agent completion and verifier success, and may
evaluate a finalization policy for workspaces that already pass verification.

## Interpretation

The corrected pilot invalidates the original conclusion that subagent
orchestration had near-zero task success. Harness behavior was a major
confounder. After the corrections, both subagent strategies solved four of
six tasks, and single-agent execution solved all six.

The rerun does not show that subagents outperform a single agent on this task
set. Single-agent execution remained more accurate, faster, and cheaper.
Parallel execution did provide a meaningful latency improvement over
sequential execution at the same pass rate, but the orchestration overhead
and additional inference workload were substantial.

## Limitations

- There is only one run per task and strategy, so no confidence intervals or
  variance estimates are available.
- The task set contains six small Python repositories and may not represent
  larger production codebases.
- Results apply to one model and one orchestration configuration.
- The corrected run intentionally changed the step limit from eight to nine
  to accommodate the reserved finalization turn.
- Provider latency and model behavior can vary over time even with
  temperature 0 and a fixed seed.

## Artifacts

- `runs.jsonl`: schema-v3 run records with bounded, redacted tool traces
- `summary.csv`: aggregate metrics by strategy
- `task-summary.csv`: metrics by task and strategy
- `comparisons.csv`: baseline-relative comparisons against single-agent runs
- `strategy-dashboard.png`: generated strategy dashboard
- `reproduce.sh`: safe reproduction script that writes to a new directory

Run `./experiments/corrected-pilot/reproduce.sh` with
`OPENROUTER_API_KEY` set to reproduce the experiment without overwriting the
published records.

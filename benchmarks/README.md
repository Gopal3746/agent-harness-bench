# Benchmark Tasks

Each benchmark task contains:

- `task.json`: the instruction, source repository, verification command,
  timeout, and classification tags.
- `repository/`: an intentionally incomplete or incorrect repository that the
  harness copies into an isolated workspace before every run.
- Tests inside the repository that provide objective pass/fail verification.

Do not fix the source repositories in place. A benchmark task must remain in
its original failing state so every agent run starts from the same snapshot.

## Included tasks

| Task | Category | Decomposable | Baseline |
|---|---|---:|---:|
| `input-validation` | Feature implementation | Yes | 5 failed, 4 passed |

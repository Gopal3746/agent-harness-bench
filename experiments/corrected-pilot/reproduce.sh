#!/usr/bin/env bash

set -euo pipefail

project_root=$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)
cd "$project_root"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running}"

result_dir=${1:-experiments/reproduced-corrected-pilot}
runs_path="$result_dir/runs.jsonl"

if [[ -s "$runs_path" ]]; then
    echo "Refusing to overwrite non-empty results: $runs_path" >&2
    exit 2
fi

mkdir -p "$result_dir"

python -m pytest
ruff check .

agent-harness benchmark-suite \
    benchmarks/tasks \
    --model minimax/minimax-m3.1-flash \
    --strategy single \
    --strategy sequential \
    --strategy parallel \
    --repetitions 1 \
    --parallelism 2 \
    --max-steps 9 \
    --temperature 0 \
    --seed 42 \
    --suite-id corrected-pilot-2026-11-10 \
    --max-total-runs 18 \
    --output "$runs_path" \
    --workspace-root workspaces/reproduced-corrected-pilot \
    --dry-run

agent-harness benchmark-suite \
    benchmarks/tasks \
    --model minimax/minimax-m3.1-flash \
    --strategy single \
    --strategy sequential \
    --strategy parallel \
    --repetitions 1 \
    --parallelism 2 \
    --max-steps 9 \
    --temperature 0 \
    --seed 42 \
    --suite-id corrected-pilot-2026-11-10 \
    --max-total-runs 18 \
    --output "$runs_path" \
    --workspace-root workspaces/reproduced-corrected-pilot

agent-harness analyze \
    "$runs_path" \
    --output "$result_dir/summary.csv" \
    --task-output "$result_dir/task-summary.csv" \
    --comparison-output "$result_dir/comparisons.csv" \
    --baseline single \
    --plot-output "$result_dir/strategy-dashboard.png"

echo "Corrected pilot artifacts written to $result_dir"

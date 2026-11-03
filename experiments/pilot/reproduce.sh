#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
PROJECT_ROOT="$(
    cd "$SCRIPT_DIR/../.."
    pwd
)"

cd "$PROJECT_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is required." >&2
    exit 1
fi

MODEL_ID="${MODEL_ID:-minimax/minimax-m3.1-flash}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/reproduction}"
SUITE_ID="${SUITE_ID:-pilot-reproduction}"

if [[ -e "$OUTPUT_DIR/runs.jsonl" ]]; then
    echo "Refusing to overwrite $OUTPUT_DIR/runs.jsonl" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

agent-harness benchmark-suite \
    benchmarks/tasks \
    --model "$MODEL_ID" \
    --strategy single \
    --strategy sequential \
    --strategy parallel \
    --repetitions 1 \
    --parallelism 2 \
    --max-steps 8 \
    --temperature 0 \
    --seed 42 \
    --suite-id "$SUITE_ID" \
    --max-total-runs 18 \
    --output "$OUTPUT_DIR/runs.jsonl" \
    --workspace-root workspaces/reproduction

agent-harness analyze \
    "$OUTPUT_DIR/runs.jsonl" \
    --output "$OUTPUT_DIR/summary.csv" \
    --task-output "$OUTPUT_DIR/task-summary.csv" \
    --comparison-output "$OUTPUT_DIR/comparisons.csv" \
    --baseline single \
    --plot-output "$OUTPUT_DIR/strategy-dashboard.png"

python - "$OUTPUT_DIR/runs.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

cost = sum(
    record.get("total_cost_usd") or 0
    for record in records
)
passed = sum(record["passed"] for record in records)
verified = sum(
    bool((record.get("verification") or {}).get("succeeded"))
    for record in records
)

print(f"runs: {len(records)}")
print(f"strict_passes: {passed}")
print(f"verification_passes: {verified}")
print(f"total_cost_usd: {cost:.6f}")
PY

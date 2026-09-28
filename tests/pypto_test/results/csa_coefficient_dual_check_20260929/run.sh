#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit the two sides together through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only the uncovered 8K/B32 dual-query path; no new DFX or Native/model comparison.
for side in baseline candidate; do
    bash "$root/run_side.sh" "$side" timing 8192 32
done

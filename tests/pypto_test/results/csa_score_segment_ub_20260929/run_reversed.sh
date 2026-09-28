#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit this timing-only order check through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Same frozen operators, reversed order, no duplicate DFX or cross-version state dump.
for side in candidate baseline; do
    bash "$root/run_reversed_side.sh" "$side" timing 131072 16
done

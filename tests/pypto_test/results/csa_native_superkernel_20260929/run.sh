#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(0 1)
    if [[ "$history" == 8192 ]]; then sides=(1 0); fi
    for super_kernel in "${sides[@]}"; do
        bash "$root/run_side.sh" "$super_kernel" "$history" "$batch"
    done
done

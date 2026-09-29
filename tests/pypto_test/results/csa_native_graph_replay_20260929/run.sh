#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for case_spec in 131072:16 8192:24; do
    bash "$root/run_side.sh" 1 "${case_spec%:*}" "${case_spec#*:}"
done

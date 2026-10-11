#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --variants ordinary wait_only set_only marked --compiled "$p/artifacts/no_successor_factorial_compile" \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_factorial_device_v1"

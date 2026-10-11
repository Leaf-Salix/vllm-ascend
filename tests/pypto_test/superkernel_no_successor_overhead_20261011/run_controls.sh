#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --compiled "$p/artifacts/no_successor_factorial_compile" \
  --variants ordinary wait_only wait_nullcheck wait_no_callback --batches 32 \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_wait_controls_device_v1"

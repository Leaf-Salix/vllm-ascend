#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
PYPTO_BOUNDARY_DIAG_BLOCKS=1 python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --compiled "$p/artifacts/no_successor_hot_wait_v2_single_compile" \
  --variants ordinary marked marked_inline_wait --batches 1 --repeats 30 --profile \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_inline_wait_single_profile"

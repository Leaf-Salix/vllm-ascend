#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
PYPTO_BOUNDARY_DIAG_BLOCKS=1 python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --compiled "$p/artifacts/no_successor_single_block_compile" \
  --batches 1 --repeats 20 --profile \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_single_block_profile_v1"

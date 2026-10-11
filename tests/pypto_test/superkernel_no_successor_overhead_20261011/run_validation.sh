#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$(python -c 'import os; print(":".join(p for p in os.environ["PYTHONPATH"].split(":") if not p.endswith(("/vllm-ascend", "/vllm"))))')"
python -m pytest tests/st/runtime/scheduling/test_task_boundary_statements.py \
  tests/st/runtime/scheduling/test_local_task_boundary_overlap.py -q -n 1 \
  --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
  --kernels-dir "$p/artifacts/hot_wait_local_device" \
  --junitxml "$p/artifacts/hot_wait_local_device.xml"
PYPTO_BOUNDARY_DIAG_BLOCKS=1 python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --compiled "$p/artifacts/no_successor_hot_wait_single_compile" \
  --variants ordinary marked_cold marked --batches 1 --repeats 30 --profile \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_hot_wait_single_profile"
PYPTO_BOUNDARY_DIAG_BLOCKS=48 python tests/st/runtime/scheduling/task_boundary_overhead_analysis.py \
  --compiled "$p/artifacts/no_successor_hot_wait_48_compile" \
  --variants ordinary marked_cold marked --batches 1 32 --repeats 30 \
  --device "$TASK_DEVICE" --output "$p/artifacts/no_successor_hot_wait_48_device"

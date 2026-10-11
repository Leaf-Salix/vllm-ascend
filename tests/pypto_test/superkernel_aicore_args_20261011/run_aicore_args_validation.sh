#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
export ASCEND_PROCESS_LOG_PATH="$p/artifacts/aicore_args_device/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_mix.py tests/st/runtime/scheduling/test_local_task_boundary_remote_dependency.py \
  tests/st/runtime/scheduling/test_task_boundary_statements.py \
  tests/st/runtime/scheduling/test_local_task_boundary_overlap.py -q -n 1 \
  --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
  --kernels-dir "$p/artifacts/aicore_args_device" \
  --junitxml "$p/artifacts/aicore_args_device.xml"

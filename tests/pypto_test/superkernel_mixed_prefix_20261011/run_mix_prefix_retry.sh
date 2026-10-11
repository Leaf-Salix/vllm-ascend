#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
export ASCEND_PROCESS_LOG_PATH="$p/artifacts/mix_prefix_device_retry/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_mix.py::test_mixed_local_boundaries \
 'tests/st/runtime/scheduling/test_local_task_boundary_remote_dependency.py::test_gated_prefix_with_unrelated_local_task[sync_start-pending]' \
 -q -n 1 --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
 --kernels-dir "$p/artifacts/mix_prefix_device_retry" --junitxml "$p/artifacts/mix_prefix_device_retry.xml"

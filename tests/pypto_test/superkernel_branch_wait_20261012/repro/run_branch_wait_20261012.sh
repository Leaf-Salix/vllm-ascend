#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/branch_wait_20261012/device"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
 tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
 -q -k 'branch_wait or default_events' --platform a2a3 --device "$TASK_DEVICE" \
 --save-kernels --kernels-dir "$output/kernels" --junitxml "$output/results.xml" \
 > "$output/run.log" 2>&1
tail -3 "$output/run.log"

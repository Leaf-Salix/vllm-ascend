#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/predicate_ready_20261012"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_predicate.py -q \
 --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$output/device" \
 --junitxml "$output/device_results.xml" > "$output/device.log" 2>&1
tail -n 3 "$output/device.log"

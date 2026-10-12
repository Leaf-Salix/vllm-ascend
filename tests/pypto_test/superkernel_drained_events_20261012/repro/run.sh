#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/drained_events_20261012/device"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_drained_events.py \
  --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$output/kernels" \
  --junitxml "$output/results.xml" -q -n 1 > "$output/run.log" 2>&1
tail -n 3 "$output/run.log"

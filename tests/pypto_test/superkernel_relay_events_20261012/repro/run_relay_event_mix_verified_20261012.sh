#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
output="$p/artifacts/relay_event_probe_20261012"
mkdir -p "$output/ascend" "$output/mix_verified"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python -m pytest \
  'tests/st/runtime/scheduling/test_local_task_boundary_mix.py::test_mixed_members_enter_prefix_before_group_completion[relay_events]' \
  -q -n 1 --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
  --kernels-dir "$output/mix_verified/kernels" --junitxml "$output/mix_verified/results.xml" \
  > "$output/mix_verified/run.log" 2>&1
tail -n 3 "$output/mix_verified/run.log"

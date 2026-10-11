#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
output="$p/artifacts/relay_event_probe_20261012"
mkdir -p "$output/ascend" "$output/mix"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/st/runtime/scheduling/task_boundary_allocated_event_probe.py \
  --event-id exhausted --compiled "$p/artifacts/relay_event_probe_compiled_20261012" \
  --output "$output" --device "$TASK_DEVICE" --work 1 --steps 64 --repeats 0 \
  > "$output/run.log" 2>&1
tail -n 3 "$output/run.log"
python -m pytest \
  'tests/st/runtime/scheduling/test_local_task_boundary_mix.py::test_mixed_members_enter_prefix_before_group_completion[relay_events]' \
  -q -n 1 --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
  --kernels-dir "$output/mix/kernels" --junitxml "$output/mix/results.xml" \
  > "$output/mix/run.log" 2>&1
tail -n 3 "$output/mix/run.log"

#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
output="$p/artifacts/blocking_event_probe_20261012"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/st/runtime/scheduling/task_boundary_allocated_event_probe.py \
  --event-id exhausted --compiled "$p/artifacts/blocking_event_probe_compiled_20261012" \
  --output "$output" --device "$TASK_DEVICE" --work 64 --steps 64 --repeats 0 \
  > "$output/run.log" 2>&1
tail -n 14 "$output/run.log"

#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
source ../env.sh
export PYTHONPATH="$SCALAR_OVERLAP_ROOT/pypto/python:$SCALAR_OVERLAP_ROOT/simpler/python:$SCALAR_OVERLAP_ROOT/simpler"
export ASCEND_PROCESS_LOG_PATH="$SCALAR_OVERLAP_ROOT/artifacts/late_arrival_alternate_timing_20261011/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$SCALAR_OVERLAP_ROOT/pypto"
python tests/st/runtime/scheduling/task_boundary_late_arrival_probe.py \
  --compiled "$SCALAR_OVERLAP_ROOT/artifacts/late_arrival_alternate_compiled_20261011" \
  --output "$SCALAR_OVERLAP_ROOT/artifacts/late_arrival_alternate_timing_20261011" \
  --repeats 15 --steps 0 --work 4 --device "$TASK_DEVICE" > "$SCALAR_OVERLAP_ROOT/artifacts/late_arrival_alternate_timing_20261011/run.log" 2>&1

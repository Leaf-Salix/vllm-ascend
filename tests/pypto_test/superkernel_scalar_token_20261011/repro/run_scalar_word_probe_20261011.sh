#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
source ../env.sh
export PYTHONPATH="$SCALAR_OVERLAP_ROOT/pypto/python:$SCALAR_OVERLAP_ROOT/simpler/python:$SCALAR_OVERLAP_ROOT/simpler"
output="$SCALAR_OVERLAP_ROOT/artifacts/scalar_word_probe_20261011"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/st/runtime/scheduling/task_boundary_scalar_token_probe.py \
 --compiled "$SCALAR_OVERLAP_ROOT/artifacts/scalar_word_probe_compiled_20261011" \
 --output "$output" --words 1 --repeats 0 --steps 0 --work 4 --device "$TASK_DEVICE" > "$output/run.log" 2>&1

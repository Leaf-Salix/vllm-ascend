#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
source ../env.sh
export PYTHONPATH="$SCALAR_OVERLAP_ROOT/pypto/python:$SCALAR_OVERLAP_ROOT/simpler/python:$SCALAR_OVERLAP_ROOT/simpler"
export ASCEND_PROCESS_LOG_PATH="$SCALAR_OVERLAP_ROOT/artifacts/compiled_poll_mix_20261011/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_mix.py -q -n 1 \
 --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
 --kernels-dir "$SCALAR_OVERLAP_ROOT/artifacts/compiled_poll_mix_20261011/kernels" \
 --junitxml "$SCALAR_OVERLAP_ROOT/artifacts/compiled_poll_mix_20261011/results.xml" \
 > "$SCALAR_OVERLAP_ROOT/artifacts/compiled_poll_mix_20261011/run.log" 2>&1

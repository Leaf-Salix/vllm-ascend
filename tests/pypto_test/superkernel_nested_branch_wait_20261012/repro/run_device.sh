#!/usr/bin/env bash
set -eo pipefail
p="${1:?pass the private scalar-overlap worktree root}"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/nested_branch_wait_repro/device"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
 tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
 -k '(local_boundary_overlap and nested_wait) or (mixed_members_enter_prefix and nested_wait)' \
 --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$output/kernels" \
 -q --junitxml "$output/results.xml" > "$output/run.log" 2>&1
tail -3 "$output/run.log"

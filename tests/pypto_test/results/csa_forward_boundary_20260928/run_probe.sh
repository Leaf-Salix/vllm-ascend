#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_forward_boundary_20260928"
mkdir -p "$root/probe/ascend"
export ASCEND_PROCESS_LOG_PATH="$root/probe/ascend"
cd "$root/probe"
python "$root/probe.py" --device "$TASK_DEVICE" --output "$root/probe" > "$root/probe/run.log" 2>&1

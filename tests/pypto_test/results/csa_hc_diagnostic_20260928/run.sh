#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
out="$repo/tests/pypto_test/results/csa_hc_diagnostic_20260928"
mkdir -p "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
cd "$out"
python "$out/diagnose.py" --source "$workspace/.cache/csa-source-baseline-2a740c1f" \
    --output "$out" --device "$TASK_DEVICE" > "$out/run.log" 2>&1

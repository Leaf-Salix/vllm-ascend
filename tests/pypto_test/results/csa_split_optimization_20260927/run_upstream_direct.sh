#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
root="$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_split_optimization_20260927"
source "$workspace/env-dsv4-0251rc1.sh"
for kind in timing swimlane; do
    output="$root/upstream_direct_cap16_h8192_b16/$kind"
    mkdir -p "$output/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/ascend"
    extra=()
    if [[ "$kind" == swimlane ]]; then extra+=(--swimlane); fi
    python "$root/upstream_probe.py" --output "$output" --upstream "$workspace/pypto-lib" \
        --device "$TASK_DEVICE" "${extra[@]}" > "$output/run.log" 2>&1
done

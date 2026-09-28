#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit this complete matrix through task-submit --device auto}"
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929
test -f "$root/source.json"
test -f "$root/parse.json"
# One allocated device for the complete matrix; no model run in this task.
case_index=0
for case_spec in 131072:4 131072:8 131072:16 131072:24 8192:24 8192:32 8192:40; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(native pto)
    if (( case_index % 2 )); then sides=(pto native); fi
    for side in "${sides[@]}" swimlane; do
        bash "$root/run_side.sh" "$side" "$history" "$batch"
    done
    case_index=$((case_index + 1))
done

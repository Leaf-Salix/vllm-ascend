#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 提交整个矩阵}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/parse.json"
index=0
for case_spec in 131072:4 131072:8 131072:16 131072:24 8192:16 8192:24 8192:32; do
    sides=(native pto)
    if (( index % 2 )); then sides=(pto native); fi
    for side in "${sides[@]}" native_incore swimlane; do
        bash "$root/run_side.sh" "$side" "${case_spec%:*}" "${case_spec#*:}"
    done
    index=$((index + 1))
done

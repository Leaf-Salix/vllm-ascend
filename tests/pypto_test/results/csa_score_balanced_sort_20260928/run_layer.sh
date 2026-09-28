#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_score_balanced_sort_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
rg -q '^ALL_COMPILE_PASS ' "$root/compile.log"
for side in baseline candidate; do
    source_repo="$workspace/.cache/csa-score-balanced-8e176285"
    extra=()
    if [[ "$side" == candidate ]]; then
        source_repo="$workspace/.cache/csa-score-balanced-sort-8e176285"
        extra=(--reference "$root/probe/baseline/pairs.pt")
    fi
    python "$root/sort_case.py" --source "$source_repo" --output "$root/probe/$side" \
        "${extra[@]}" > "$root/probe/$side/run.log" 2>&1
done
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$root" "$workspace/.cache/csa-score-balanced-sort-8e176285" 16 \
    "$workspace/.cache/csa-score-balanced-8e176285"

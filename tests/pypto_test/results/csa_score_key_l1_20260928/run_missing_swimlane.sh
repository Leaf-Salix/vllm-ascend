#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_score_key_l1_20260928"
rg -q '^COMPILE_PASS ' "$root/compile.log"
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-ub-combined-2d2f9ca0"
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-score-key-l1-2d2f9ca0"
    fi
    out="$root/h8192_b16/swimlane/$label"
    [[ ! -e "$out/report.json" ]]
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 16 --history 8192 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 0 --swimlane --swimlane-graph --swimlane-windows 4 \
        > "$out/run.log" 2>&1
done

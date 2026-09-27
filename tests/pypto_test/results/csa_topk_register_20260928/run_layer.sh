#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_topk_register_20260928"
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-forward-boundary-71153bb3"
    extra=()
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-topk-register-71153bb3"
        extra=(--graph)
    fi
    out="$root/accuracy/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 8 --history 131072 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --save-state --timing-iters 20 --timing-warmup 5 \
        --timing-metadata reuse "${extra[@]}" > "$out/run.log" 2>&1
done
python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
    --root "$root/accuracy" --scope "71153bb3 vs UB Top-K root, B8/H131072/S6, fixed order, variable scale; not Native equivalence"
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-forward-boundary-71153bb3"
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-topk-register-71153bb3"
    fi
    out="$root/swimlane/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 8 --history 131072 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --swimlane --swimlane-graph --swimlane-windows 4 \
        > "$out/run.log" 2>&1
done

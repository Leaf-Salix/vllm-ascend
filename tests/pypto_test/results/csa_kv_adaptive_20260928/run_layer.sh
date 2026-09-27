#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_kv_adaptive_20260928"
for shape in 8192:16 8192:40 8192:4; do
    history="${shape%:*}"
    batch="${shape#*:}"
    timing=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
    if [[ "$batch" == 4 ]]; then timing=(); fi
    for label in baseline candidate; do
        source_repo="$workspace/.cache/csa-no-seed-71153bb3"
        extra=()
        if [[ "$label" == candidate ]]; then
            source_repo="$workspace/.cache/csa-kv-adaptive-f1e4cee2"
            extra=(--graph)
        fi
        out="$root/h${history}_b${batch}/$label"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
            --atomic-add 0 --deterministic-level 1 --save-state "${timing[@]}" "${extra[@]}" > "$out/run.log" 2>&1
    done
    python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
        --root "$root/h${history}_b${batch}" \
        --scope "f1e4cee2 vs adaptive fixed-K KV M32/M64, H${history}/B${batch}/S6, atomic0; not Native equivalence"
done
history=8192
batch=40
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-no-seed-71153bb3"
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-kv-adaptive-f1e4cee2"
    fi
    out="$root/h${history}_b${batch}/swimlane/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --swimlane --swimlane-graph --swimlane-windows 4 \
        > "$out/run.log" 2>&1
done

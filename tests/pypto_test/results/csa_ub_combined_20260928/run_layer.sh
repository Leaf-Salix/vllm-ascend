#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_ub_combined_20260928"
for history in 131072 8192; do
    for phase in timing swimlane; do
        labels=(candidate baseline)
        if [[ "$history" == 8192 ]]; then labels=(baseline candidate); fi
        for label in "${labels[@]}"; do
            source_repo="$workspace/.cache/csa-ascendc-topk-hc-ep16-d1f170ff"
            extra=()
            if [[ "$label" == candidate ]]; then
                source_repo="$workspace/.cache/csa-ub-combined-2d2f9ca0"
                extra=(--graph)
            fi
            if [[ "$phase" == timing ]]; then
                out="$root/h${history}_b16/$label"
                extra+=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse --save-state)
            else
                out="$root/h${history}_b16/swimlane/$label"
                extra=(--swimlane --swimlane-graph --swimlane-windows 2)
            fi
            mkdir -p "$out/ascend"
            export ASCEND_PROCESS_LOG_PATH="$out/ascend"
            cd "$out"
            python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
                --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
                --output "$out" --device "$TASK_DEVICE" --batch 16 --history "$history" \
                --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
                --atomic-add 0 --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
        done
    done
done

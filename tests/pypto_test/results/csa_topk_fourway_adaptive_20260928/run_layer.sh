#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_topk_fourway_adaptive_20260928"
for history in 8192 131072; do
    for phase in timing swimlane; do
        # 两档交换先后顺序，保留各自Native控制和全部原始样本。
        labels=(baseline candidate)
        if [[ "$history" == 8192 ]]; then labels=(candidate baseline); fi
        for label in "${labels[@]}"; do
            source_repo="$workspace/.cache/csa-kv-k512-ep16-d950ba3d"
            extra=()
            if [[ "$label" == candidate ]]; then
                source_repo="$workspace/.cache/csa-topk-fourway-adaptive-e58ddc94"
                extra=(--graph)
            fi
            if [[ "$phase" == timing ]]; then
                out="$root/h${history}_b16/$label"
                extra+=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse --save-state)
            else
                out="$root/h${history}_b16/swimlane/$label"
                extra=(--swimlane --swimlane-graph --swimlane-windows 4)
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

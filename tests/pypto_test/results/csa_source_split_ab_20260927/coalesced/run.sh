#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
result="$repo/tests/pypto_test/results/csa_source_split_ab_20260927/coalesced"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
for history in 8192 131072; do
    for layout in original coalesced; do
        source_repo="$workspace/.cache/csa-source-baseline-2a740c1f"
        extra=()
        if [[ "$layout" == coalesced ]]; then
            source_repo="$workspace/.cache/csa-source-coalesced-2a740c1f"
            extra=(--graph)
        fi
        out="$result/accuracy/h${history}_b4/$layout"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$result/../contiguous_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch 4 --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
            --atomic-add 0 --deterministic-level 1 --save-state "${extra[@]}" \
            > "$out/run.log" 2>&1
    done
    python "$result/compare_accuracy.py" "$history"
done
for history in 8192 131072; do
    # Reverse the order for the second context; do not pick fastest samples.
    layouts=(original coalesced)
    if [[ "$history" == 131072 ]]; then layouts=(coalesced original); fi
    for layout in "${layouts[@]}"; do
        source_repo="$workspace/.cache/csa-source-baseline-2a740c1f"
        if [[ "$layout" == coalesced ]]; then
            source_repo="$workspace/.cache/csa-source-coalesced-2a740c1f"
        fi
        out="$result/timing/h${history}_b16/$layout"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$result/../contiguous_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch 16 --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
            --atomic-add 1 --deterministic-level 0 \
            --timing-iters 20 --timing-warmup 5 --timing-metadata reuse \
            > "$out/run.log" 2>&1
    done
done

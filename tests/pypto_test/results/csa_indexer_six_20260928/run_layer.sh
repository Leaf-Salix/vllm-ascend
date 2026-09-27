#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_indexer_six_20260928"
for case_name in baseline_short triple_short triple_long six_long; do
    case "$case_name" in
        baseline_short) revision=source-baseline; history=8192 ;;
        triple_short) revision=indexer-triple; history=8192 ;;
        triple_long) revision=indexer-triple; history=131072 ;;
        six_long) revision=indexer-six; history=131072 ;;
    esac
    source_repo="$workspace/.cache/csa-${revision}-2a740c1f"
    kinds=(timing)
    if [[ "$case_name" == six_long ]]; then
        kinds+=(swimlane)
    fi
    for kind in "${kinds[@]}"; do
        out="$root/$case_name/$kind"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        if [[ "$kind" == timing ]]; then
            extra=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
        else
            extra=(--swimlane --swimlane-graph --swimlane-windows 4)
        fi
        python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch 16 --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
            --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
    done
done

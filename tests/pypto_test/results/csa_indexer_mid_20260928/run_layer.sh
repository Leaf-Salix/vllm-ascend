#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_indexer_mid_20260928"
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-indexer-six-2a740c1f"
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-indexer-mid-9a01a276"
    fi
    for kind in timing swimlane; do
        out="$root/$label/$kind"
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
            --output "$out" --device "$TASK_DEVICE" --batch 8 --history 131072 \
            --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
            --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
    done
done

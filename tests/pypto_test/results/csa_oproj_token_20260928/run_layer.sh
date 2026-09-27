#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-oproj-token-2a740c1f"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_oproj_token_20260928"
for label in timing_b16 graph_b3; do
    out="$root/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    if [[ "$label" == timing_b16 ]]; then
        extra=(--batch 16 --atomic-add 1 --deterministic-level 0 --timing-iters 20 --timing-warmup 5)
    else
        extra=(--batch 3 --atomic-add 0 --deterministic-level 1 --graph)
    fi
    python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --history 8192 \
        --layer-index 4 --variant performance --weight-nz-mode 2 "${extra[@]}" \
        > "$out/run.log" 2>&1
done

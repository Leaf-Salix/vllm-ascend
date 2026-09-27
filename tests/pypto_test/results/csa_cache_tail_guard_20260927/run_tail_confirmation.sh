#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
for variant in baseline candidate; do
    source_repo="$repo"
    if [[ "$variant" == baseline ]]; then
        source_repo="$workspace/.cache/csa-cache-05cb2758"
    fi
    output="$repo/tests/pypto_test/results/csa_cache_tail_guard_20260927/confirmation/$variant"
    mkdir -p "$output/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/ascend"
    cd "$output"
    python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$output" --device "$TASK_DEVICE" --batch 16 --history 131072 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
        --deterministic-level 0 --timing-iters 100 --timing-warmup 5 \
        --timing-metadata reuse > "$output/run.log" 2>&1
done

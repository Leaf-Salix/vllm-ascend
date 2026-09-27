#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-source-baseline-2a740c1f"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_duplicate_requests_20260928"
for atomic in 1 0; do
    out="$root/atomic$atomic"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$root/duplicate_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 16 --history 8192 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add "$atomic" \
        --deterministic-level 0 --timing-iters 10 --timing-warmup 3 \
        > "$out/run.log" 2>&1
done

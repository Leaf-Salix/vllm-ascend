#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-forward-boundary-71153bb3"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_atomic_current_20260928"
for shape in long short; do
    case "$shape" in
        long) batch=8; history=131072 ;;
        short) batch=16; history=8192 ;;
    esac
    for atomic in 1 0; do
        out="$root/single/$shape/atomic$atomic"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add "$atomic" \
            --deterministic-level 0 --timing-iters 20 --timing-warmup 5 --timing-metadata reuse \
            > "$out/run.log" 2>&1
    done
done

#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_b24_cann92_20260928"
source_repo="$workspace/.cache/csa-b24-cann92-3b27c7fd"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PTO_CSA_VARIANT=pkg:dsv4_csa_b24_cann92_3b27c7fd
export PTO_CSA_RING_HEAP_MB=256,128,256,32
export PTO_CSA_RING_TASK_WINDOW=4096
rg -q 'COMPILE_PASS' "$root/prepare.json"
for phase in timing swimlane; do
    out="$root/$phase"
    [[ ! -e "$out/report.json" ]]
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    if [[ "$phase" == timing ]]; then
        extra=(--graph --timing-iters 20 --timing-warmup 5 --timing-metadata reuse --profile)
    else
        extra=(--swimlane --swimlane-graph --swimlane-windows 4)
    fi
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 24 --history 131072 \
        --layer-index 4 --variant "$PTO_CSA_VARIANT" --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
done

#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_qr_ub_20260928/tail_b10"
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-ascendc-topk-hc-ep16-d1f170ff"
    extra=()
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-qr-ub-a66255ea"
        extra=(--graph)
    fi
    out="$root/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 10 --history 8192 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --save-state "${extra[@]}" > "$out/run.log" 2>&1
done
python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
    --root "$root" --scope 'QR input/gamma UB reuse; B10/S6/T60 tail; fixed reduction; no timing or EP16 claim'

#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_hc_input_rms_20260928/tail_b10"
for variant in performance precision; do
    for label in baseline candidate; do
        source_repo="$workspace/.cache/csa-kv-k512-ep16-d950ba3d"
        extra=()
        if [[ "$label" == candidate ]]; then
            source_repo="$workspace/.cache/csa-hc-input-rms-9edb8dfe"
            extra=(--graph)
        fi
        out="$root/$variant/$label"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch 10 --history 8192 \
            --layer-index 4 --variant "$variant" --weight-nz-mode 2 --seed 1024 \
            --atomic-add 0 --deterministic-level 1 --save-state "${extra[@]}" > "$out/run.log" 2>&1
    done
    python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
        --root "$root/$variant" \
        --scope "HC widen/RMS fusion and shared helper extraction; B10/S6/T60 tail, variant=$variant; not Native equivalence"
done

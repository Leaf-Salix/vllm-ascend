#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_qr_k512_20260928"
for shape in 131072:16 8192:16; do
    history="${shape%:*}"
    batch="${shape#*:}"
    for phase in timing swimlane; do
        for label in baseline candidate; do
            source_repo="$workspace/.cache/csa-kv-k512-ep16-d950ba3d"
            extra=()
            if [[ "$label" == candidate ]]; then
                source_repo="$workspace/.cache/csa-qr-k512-e58ddc94"
                extra=(--graph)
            fi
            if [[ "$phase" == timing ]]; then
                out="$root/h${history}_b${batch}/$label"
                extra+=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse --save-state)
            else
                out="$root/h${history}_b${batch}/swimlane/$label"
                extra=(--swimlane --swimlane-graph --swimlane-windows 2)
            fi
            mkdir -p "$out/ascend"
            export ASCEND_PROCESS_LOG_PATH="$out/ascend"
            cd "$out"
            python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
                --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
                --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
                --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
                --atomic-add 0 --deterministic-level 1 "${extra[@]}" > "$out/run.log" 2>&1
        done
        if [[ "$phase" == timing ]]; then
            python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
                --root "$root/h${history}_b${batch}" \
                --scope "e58ddc94 vs fixed-K QR K512, H${history}/B${batch}/S6, atomic0; not Native equivalence"
        fi
    done
done

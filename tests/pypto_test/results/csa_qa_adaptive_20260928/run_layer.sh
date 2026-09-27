#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_qa_adaptive_20260928"
for shape in 8192:16 8192:40 131072:16 8192:4; do
    history="${shape%:*}"
    batch="${shape#*:}"
    timing=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
    if [[ "$batch" == 4 ]]; then timing=(); fi
    for label in baseline candidate; do
        source_repo="$workspace/.cache/csa-kv-adaptive-f1e4cee2"
        extra=()
        if [[ "$label" == candidate ]]; then
            source_repo="$workspace/.cache/csa-qa-adaptive-88d0744f"
            extra=(--graph)
        fi
        out="$root/h${history}_b${batch}/$label"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
            --atomic-add 0 --deterministic-level 1 --save-state "${timing[@]}" "${extra[@]}" > "$out/run.log" 2>&1
    done
    python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
        --root "$root/h${history}_b${batch}" \
        --scope "88d0744f vs adaptive fixed-K QR M32/M64, H${history}/B${batch}/S6, atomic0; not Native equivalence"
done
# Reuse the same 88d0744f baseline's recent long/B40 DFX; only short B16 lacks it.
for shape in 8192:16 8192:40 131072:16; do
    history="${shape%:*}"
    batch="${shape#*:}"
    for label in baseline candidate; do
        if [[ "$label" == baseline && "$shape" != 8192:16 ]]; then continue; fi
        source_repo="$workspace/.cache/csa-kv-adaptive-f1e4cee2"
        if [[ "$label" == candidate ]]; then
            source_repo="$workspace/.cache/csa-qa-adaptive-88d0744f"
        fi
        out="$root/h${history}_b${batch}/swimlane/$label"
        mkdir -p "$out/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/ascend"
        cd "$out"
        python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
            --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
            --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
            --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
            --atomic-add 0 --deterministic-level 1 --swimlane --swimlane-graph --swimlane-windows 2 \
            > "$out/run.log" 2>&1
    done
done

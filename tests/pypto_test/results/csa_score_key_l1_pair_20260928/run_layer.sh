#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
[[ "$(task-submit --status task_20260928_182811_290034987)" == 'completed (exit=0)' ]]
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928"
rg -q '^COMPILE_PASS ' "$root/compile.log"
# Only long small-batch changes; use B4 to exercise it and short B16 as control.
for case_spec in 131072:4 8192:16; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    for phase in timing swimlane; do
        labels=(candidate baseline)
        if [[ "$history" == 8192 ]]; then labels=(baseline candidate); fi
        for label in "${labels[@]}"; do
            source_repo="$workspace/.cache/csa-key-l1-seven-554b3bca"
            extra=()
            if [[ "$label" == candidate ]]; then
                source_repo="$workspace/.cache/csa-score-key-l1-pair-554b3bca"
                extra=(--graph)
            fi
            if [[ "$phase" == timing ]]; then
                out="$root/h${history}_b${batch}/$label"
                extra+=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse --save-state)
            else
                out="$root/h${history}_b${batch}/swimlane/$label"
                extra=(--swimlane --swimlane-graph --swimlane-windows 4)
            fi
            [[ ! -e "$out/report.json" ]]
            mkdir -p "$out/ascend"
            export ASCEND_PROCESS_LOG_PATH="$out/ascend"
            cd "$out"
            python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
                --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
                --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
                --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
                --atomic-add 0 --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
        done
    done
done

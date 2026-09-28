#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-key-l1-seven-554b3bca"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_key_l1_seven_20260928"
# The two B16 cases already tested this exact operator implementation.
# Cover only the five remaining cases before scheduling the model sweep.
for case_spec in 131072:4 131072:8 8192:24 8192:32 8192:40; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    for phase in timing swimlane; do
        if [[ "$phase" == timing ]]; then
            out="$root/layer/h${history}_b${batch}"
            extra=(--graph --timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
        else
            out="$root/layer/h${history}_b${batch}/swimlane"
            extra=(--swimlane --swimlane-graph --swimlane-windows 2)
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

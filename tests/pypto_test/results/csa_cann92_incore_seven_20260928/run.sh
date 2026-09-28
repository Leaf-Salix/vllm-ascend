#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit this script through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-cann92-baseline-e33d842a"
root="$repo/tests/pypto_test/results/csa_cann92_incore_seven_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
# One frozen retained operator, one allocated card; profiling follows timing.
# Neither rejected max-length reuse nor query-sort looping is in this source.
for case_spec in 131072:4 131072:8 131072:16 8192:16 8192:24 8192:32 8192:40; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    for phase in timing swimlane; do
        if [[ "$phase" == timing ]]; then
            out="$root/layer/h${history}_b${batch}"
            extra=(--graph --timing-iters 20 --timing-warmup 5 --timing-metadata reuse --profile)
        else
            out="$root/layer/h${history}_b${batch}/swimlane"
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

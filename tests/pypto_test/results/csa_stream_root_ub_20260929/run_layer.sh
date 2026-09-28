#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
# Check queue handles before submission. Even read-only task-submit --status
# is rejected inside this daemon's task execution environment.
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_stream_root_ub_20260929"
candidate_repo="$workspace/.cache/csa-stream-root-ub-67a14117-candidate"
baseline_repo="$workspace/.cache/csa-stream-root-ub-67a14117-baseline"
export PTO_CSA_VARIANT=pkg:dsv4_csa_stream_root_ub_67a14117
export PTO_CSA_RING_HEAP_MB=256,128,256,32
export PTO_CSA_RING_TASK_WINDOW=4096
for side in baseline candidate; do
    rg -q 'COMPILE_PASS' "$root/compile_${side}.json"
done
csa_case_specs=(131072:16 8192:24)
for case_spec in "${csa_case_specs[@]}"; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    for phase in timing swimlane; do
        labels=(candidate baseline)
        if [[ "$history" == 8192 ]]; then labels=(baseline candidate); fi
        for label in "${labels[@]}"; do
            source_repo="$baseline_repo"
            extra=()
            if [[ "$label" == candidate ]]; then
                source_repo="$candidate_repo"
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
                --layer-index 4 --variant "$PTO_CSA_VARIANT" --weight-nz-mode 2 --seed 1024 \
                --atomic-add 0 --deterministic-level 0 "${extra[@]}" > "$out/run.log" 2>&1
        done
    done
done

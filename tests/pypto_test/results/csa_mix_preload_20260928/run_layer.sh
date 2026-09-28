#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
: "${1:?Pass the frozen CSA source directory selected after the HC experiment}"
source_repo="$1"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_mix_preload_20260928"
original_python="$workspace/.venv-dsv4-0251rc1/bin/python"
candidate_python="$workspace/.cache/pypto-pr2389-3e87a843/.venv/bin/python"
for history in 131072 8192; do
    labels=(original gate0 gate50)
    if [[ "$history" == 8192 ]]; then labels=(original gate50 gate0); fi
    for phase in timing swimlane; do
        for label in "${labels[@]}"; do
            if [[ "$phase" == swimlane && "$label" == original ]]; then continue; fi
            python_bin="$candidate_python"
            wrapper="$root/isolated_case.py"
            export SIMPLER_MIX_PRELOAD_MAX_REMAINING_US=50
            if [[ "$label" == original ]]; then
                python_bin="$original_python"
                wrapper="$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py"
                unset SIMPLER_MIX_PRELOAD_MAX_REMAINING_US
            elif [[ "$label" == gate0 ]]; then
                export SIMPLER_MIX_PRELOAD_MAX_REMAINING_US=0
            fi
            if [[ "$phase" == timing ]]; then
                out="$root/h${history}_b16/$label"
                extra=(--timing-iters 50 --timing-warmup 5 --timing-metadata reuse --save-state)
                if [[ "$label" == gate50 ]]; then extra+=(--graph); fi
            else
                out="$root/h${history}_b16/swimlane/$label"
                extra=(--swimlane --swimlane-graph --swimlane-windows 2)
            fi
            mkdir -p "$out/ascend"
            export ASCEND_PROCESS_LOG_PATH="$out/ascend"
            cd "$out"
            "$python_bin" "$wrapper" "$source_repo" \
                --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
                --output "$out" --device "$TASK_DEVICE" --batch 16 --history "$history" \
                --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
                --atomic-add 0 --deterministic-level 1 "${extra[@]}" > "$out/run.log" 2>&1
        done
        if [[ "$phase" == timing ]]; then
            for baseline in original gate0; do
                comparison="$root/h${history}_b16/${baseline}_vs_gate50"
                mkdir -p "$comparison"
                ln -s "../$baseline" "$comparison/baseline"
                ln -s ../gate50 "$comparison/candidate"
                "$original_python" "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
                    --root "$comparison" \
                    --scope "Same CSA, $baseline vs Simpler #2389 gate50, H${history}/B16/S6, atomic0; not Native equivalence"
            done
        fi
    done
done

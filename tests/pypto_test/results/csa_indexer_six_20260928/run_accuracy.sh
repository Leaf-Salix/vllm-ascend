#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_indexer_six_20260928/accuracy"
# H32768 + S6 creates a full 8192-row leaf and a causal tail leaf,
# selecting the same long-history grouped kernel with less state to save.
for label in baseline candidate; do
    source_repo="$workspace/.cache/csa-source-baseline-2a740c1f"
    extra=()
    if [[ "$label" == candidate ]]; then
        source_repo="$workspace/.cache/csa-indexer-six-2a740c1f"
        extra=(--graph)
    fi
    out="$root/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 16 --history 32768 \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --save-state "${extra[@]}" > "$out/run.log" 2>&1
done
python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py"

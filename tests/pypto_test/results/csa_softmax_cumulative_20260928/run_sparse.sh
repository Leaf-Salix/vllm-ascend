#!/usr/bin/env bash
set -eo pipefail
[[ "${TASK_DEVICE:?Submit through task-submit --device 0}" == 0 ]]
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_softmax_cumulative_20260928"
input="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_pmu/native/native_sparse.pt"
for label in baseline cumulative tail_b3; do
    source_repo="$workspace/.cache/csa-softmax-cumulative-2a740c1f"
    args=(--input "$input")
    if [[ "$label" == baseline ]]; then
        source_repo="$workspace/.cache/csa-source-baseline-2a740c1f"
    elif [[ "$label" == tail_b3 ]]; then
        args=(--synthetic-batch 3)
    fi
    out="$root/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$source_repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
        "${args[@]}" --output "$out" --variant performance > "$out/run.log" 2>&1
done

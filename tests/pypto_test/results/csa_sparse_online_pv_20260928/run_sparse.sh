#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_sparse_online_pv_20260928"
source_repo="$workspace/.cache/csa-sparse-online-pv-a66255ea"
input="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_pmu/native/native_sparse.pt"
for label in fixed_native tail_b3; do
    args=(--input "$input")
    if [[ "$label" == tail_b3 ]]; then args=(--synthetic-batch 3); fi
    out="$root/$label"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$root/sparse_case.py" "$source_repo" "${args[@]}" \
        --output "$out" --variant performance --device "$TASK_DEVICE" > "$out/run.log" 2>&1
done
python "$root/compare_sparse.py"

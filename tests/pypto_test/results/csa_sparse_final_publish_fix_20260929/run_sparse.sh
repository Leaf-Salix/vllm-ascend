#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 提交}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
root="$repo/tests/pypto_test/results/csa_sparse_final_publish_fix_20260929"
input="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_pmu/native/native_sparse.pt"
export PTO_CSA_VARIANT=pkg:dsv4_csa_sparse_final_publish_4ffccb7b
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
test -f "$root/compile_candidate.json"
for side in baseline broken candidate; do
    source_repo="$workspace/.cache/csa-sparse-final-publish-fix-4ffccb7b-$side"
    if [[ "$side" == broken ]]; then
        source_repo="$workspace/.cache/csa-sparse-final-publish-4ffccb7b-candidate"
    fi
    out="$root/sparse/$side"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$root/sparse_case.py" "$source_repo" --input "$input" \
        --output "$out" --variant performance --device "$TASK_DEVICE" > "$out/run.log" 2>&1
done
python "$root/compare_sparse.py"

#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_full_valid_no_zero"
mkdir -p "$root/tail_b3_v2/ascend"
export ASCEND_PROCESS_LOG_PATH="$root/tail_b3_v2/ascend"
cd "$root/tail_b3_v2"
python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" --synthetic-batch 3 --variant performance --output "$root/tail_b3_v2" > run.log 2>&1
cd "$repo"
for history in 8192 131072; do
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh incore_sparse_full_valid_no_zero_v2 "$history" 16 timing
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh incore_sparse_full_valid_no_zero_v2 "$history" 16 swimlane
done

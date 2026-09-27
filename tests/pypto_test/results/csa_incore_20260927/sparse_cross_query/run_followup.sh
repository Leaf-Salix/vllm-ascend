#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
case_dir="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_cross_query/zero_work_cores_b3"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
cd "$case_dir"
python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
    --synthetic-batch 3 --variant performance --output "$case_dir" > run.log 2>&1
cd "$repo"
for history in 8192 131072; do
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_cross_query "$history" 16 timing
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_cross_query "$history" 16 swimlane
done

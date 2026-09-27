#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
source "$workspace/env-dsv4-0251rc1.sh"
cd "$workspace/vllm-ascend-dsv4-pto-0251rc1"
result_dir="tests/pypto_test/results/csa_split_optimization_20260927/${1:-fixpipe_mat}"
mkdir -p "$result_dir"
python -m pytest -q tests/pypto_test/test_csa_indexer_cube.py > "$result_dir/score_test.log" 2>&1

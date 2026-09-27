#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
python tests/pypto_test/test_csa_indexer_cube.py > tests/pypto_test/results/csa_split_optimization_20260927/v8_native_pair/tile_test.log 2>&1
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v8_native_pair 131072 16 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v8_native_pair 131072 16 swimlane

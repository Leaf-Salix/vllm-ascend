#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for batch in 24 32; do
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early 8192 "$batch" timing
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early 8192 "$batch" swimlane
done

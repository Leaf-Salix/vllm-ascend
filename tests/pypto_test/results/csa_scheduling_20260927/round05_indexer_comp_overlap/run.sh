#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for batch in 16 40; do
for kind in timing swimlane; do
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh round05_indexer_comp_overlap 8192 "$batch" "$kind"
done
done

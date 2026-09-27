#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for history in 8192 131072; do
    for kind in timing swimlane; do
        bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh incore_indexer_score_panel1024 "$history" 16 "$kind"
    done
done

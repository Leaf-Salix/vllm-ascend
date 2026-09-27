#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
# Completes the same da2e2368 operator matrix; no historical source switching.
for pair in '131072 4' '131072 8' '8192 24' '8192 32'; do
    read -r history batch <<< "$pair"
    for kind in timing swimlane; do
        bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh \
            sparse_cross_query "$history" "$batch" "$kind"
    done
done

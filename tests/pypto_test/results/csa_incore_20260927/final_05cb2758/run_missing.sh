#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
runner=tests/pypto_test/results/csa_incore_20260927/final_05cb2758/run_case.sh
for batch in 4 8; do
    bash "$runner" 131072 "$batch" timing
    bash "$runner" 131072 "$batch" swimlane
done
for batch in 24 32 40; do
    bash "$runner" 8192 "$batch" timing
    bash "$runner" 8192 "$batch" swimlane
done

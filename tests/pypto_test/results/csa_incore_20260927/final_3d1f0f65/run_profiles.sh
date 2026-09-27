#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
runner=tests/pypto_test/results/csa_incore_20260927/final_3d1f0f65/run_case.sh
bash "$runner" 8192 16 profile
bash "$runner" 131072 16 profile

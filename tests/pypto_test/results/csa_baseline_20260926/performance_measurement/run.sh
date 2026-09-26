#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 提交}"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source "$repo_root/../env-dsv4-0251rc1.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export PYTHONPATH="$repo_root/tests/pypto_test:${PYTHONPATH:-}"
python "$repo_root/tests/pypto_test/results/csa_baseline_20260926/performance_measurement/single_card.py"

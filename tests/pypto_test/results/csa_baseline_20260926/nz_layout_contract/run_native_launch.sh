#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
cd /data/pyptouser/qinchuanyu/pto-eager/pypto
source .claude/skills/testing/load-env.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_baseline_20260926/nz_layout_contract/ascend
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
exec python -m pytest -q tests/st/runtime/kernel/test_native_nz.py --platform a2a3 --device 0 --junitxml=../vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_baseline_20260926/nz_layout_contract/native_launch.xml

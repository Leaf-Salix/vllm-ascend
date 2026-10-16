#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012
source /data/pyptouser/qinchuanyu/pto-eager/pypto/.claude/skills/testing/load-env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
CSAJIT_OUT="$PWD/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/jit_group_inline_device_v1"
mkdir -p "$CSAJIT_OUT/ascend"
export ASCEND_PROCESS_LOG_PATH="$CSAJIT_OUT/ascend"
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py tests/pypto_test/csa_static_spmd_feasibility_20261015/test_jit_group_inline_device.py --device "$TASK_DEVICE" --output "$CSAJIT_OUT"

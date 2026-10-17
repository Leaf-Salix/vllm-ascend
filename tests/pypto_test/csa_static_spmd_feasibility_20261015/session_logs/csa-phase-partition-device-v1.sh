#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CSA_STATIC_PYPTO_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012
CSAJIT_OUT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/phase_partition_device_v1
mkdir -p "$CSAJIT_OUT/ascend"
export ASCEND_PROCESS_LOG_PATH="$CSAJIT_OUT/ascend"
python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/csa_static_spmd_feasibility_20261015/test_phase_partition_device.py --kernel /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/phase_partition_PYPTO_fix_v2/kernel --device "$TASK_DEVICE" --output "$CSAJIT_OUT"


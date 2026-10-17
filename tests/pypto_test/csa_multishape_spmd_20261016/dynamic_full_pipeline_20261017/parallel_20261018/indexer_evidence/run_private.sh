#!/usr/bin/env bash
# 只选已有隔离依赖；真正使用NPU时由被调用脚本核对TASK_DEVICE。
set -eo pipefail
csa_root=/data/pyptouser/qinchuanyu/pto-eager
source "$csa_root/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$csa_root/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$csa_root/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_ASCEND_ENABLE_NZ=2
python "$csa_root/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" "$@"

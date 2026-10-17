#!/usr/bin/env bash
# 与主reproduce_cpu.sh相同的既有隔离依赖，只换测试hook包装入口。
set -eo pipefail
csa_evidence=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
csa_workspace=/data/pyptouser/qinchuanyu/pto-eager
source "$csa_workspace/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$csa_workspace/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$csa_workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
python "$csa_workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$csa_evidence/cpu_probe.py" --output "$csa_evidence/cpu_v1" \
  --attention-query-metadata off --spmd-source "$csa_evidence/source_snapshot"

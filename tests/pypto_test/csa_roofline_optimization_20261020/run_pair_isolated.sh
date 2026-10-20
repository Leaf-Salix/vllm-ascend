#!/usr/bin/env bash
# 成对压缩专用工具链。旧v5任务继续使用原run_isolated.sh，不改变其版本。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-pair-compact-20261020"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$repo/tests/pypto_test/csa_critical_path_20261020/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
unset HC_PRE_TEST_OPERATOR_SOURCE
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT" \
  --ptoas-root "$workspace/.worktrees/ptoas-csa-pair-compact-20261020/build" "$@"

#!/usr/bin/env bash
# 本轮只读绑定已验证复现工具链；不装载未验证的 SDMA 修复草稿。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-topk-compact-20261020"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$repo/tests/pypto_test/csa_critical_path_20261020/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
unset HC_PRE_TEST_OPERATOR_SOURCE
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT" \
  --ptoas-root "$workspace/.worktrees/ptoas-csa-topk-compact-20261020/build" "$@"

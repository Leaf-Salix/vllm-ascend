#!/usr/bin/env bash
# 测试进程绑定新版 SDMA 工具链；仍使用既有已验证 PTOAS，不改变共享环境。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-sdma-refresh-20261020"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-sdma-refresh-20261020"
export PYTHONPATH="$repo/tests/pypto_test/csa_sdma_prefetch_20261019/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH="$here/device_logs"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT" \
    --ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"

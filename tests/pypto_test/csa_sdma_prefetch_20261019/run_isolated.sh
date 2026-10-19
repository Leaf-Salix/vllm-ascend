#!/usr/bin/env bash
# 只绑定本轮工具链；NPU 入口在各 Python 脚本中检查 TASK_DEVICE。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-sdma-20261019"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-sdma-20261019"
export PYTHONPATH="$here/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH="$here/device_logs"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
toolchain="$repo/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_key_l1_region_sync_20261019/toolchain_v3"
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT" \
  --ptoas-root "$toolchain" "$@"

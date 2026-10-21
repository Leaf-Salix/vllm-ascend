#!/usr/bin/env bash
# 四路共同使用已验证的成对压缩组件，不改共享安装或既有Top-K复现入口。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-pair-compact-20261020"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export CSA_STATIC_PYPTO_ROOT="$CSA_SDMA_PYPTO_ROOT" CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$CSA_SDMA_SIMPLER_ROOT"
export PYTHONPATH="$repo/tests/pypto_test/csa_critical_path_20261020/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
unset HC_PRE_TEST_OPERATOR_SOURCE
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT" \
    --ptoas-root "$workspace/.worktrees/ptoas-csa-pair-compact-20261020/build" \
    "$here/run_device.py" "$@"

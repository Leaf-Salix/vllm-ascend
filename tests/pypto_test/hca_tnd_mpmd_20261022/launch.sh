#!/usr/bin/env bash
# CPU 编译不占设备；真机调用由 task-submit 分配。SDK 只选公共清单。
set -eo pipefail
hca_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hca_repo=$(cd -- "$hca_here/../../.." && pwd)
hca_workspace=$(dirname -- "$hca_repo")
source "$hca_workspace/env-dsv4-0251rc1.sh"
source "$hca_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
hca_runner="$hca_here/run_case.py"
if [[ "${1:-}" == --experiment ]]; then
    hca_experiment=${2:?冻结实验目录}
    shift 2
    hca_runner="$hca_experiment/run_case.py"
    set -- --source "$hca_experiment/operator" "$@"
elif [[ "${1:-}" == --script ]]; then
    hca_runner=${2:?独立测试脚本}
    shift 2
fi
exec python "$hca_repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$hca_workspace" \
    --vllm-root "$hca_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$hca_runner" "$@"

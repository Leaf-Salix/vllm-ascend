#!/usr/bin/env bash
# Native专用：静态编译、SuperKernel、NZ与ACLGraph；每任务独享可写OPP。
set -eo pipefail
: "${TASK_DEVICE:?必须经task-submit分配设备}"
hca_spec_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hca_spec_repo=$(cd -- "$hca_spec_here/../../.." && pwd)
hca_spec_workspace=$(dirname -- "$hca_spec_repo")
source "$hca_spec_workspace/env-dsv4-0251rc1.sh"
source "$hca_spec_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
hca_spec_output=${1:?新的结果目录}
shift
if [[ -e "$hca_spec_output" ]]; then
    printf '禁止覆盖结果目录：%s\n' "$hca_spec_output" >&2
    exit 2
fi
mkdir -p "$hca_spec_output/ascend"
export ASCEND_PROCESS_LOG_PATH="$hca_spec_output/ascend"
ASCEND_OPP_PATH=$(python "$hca_spec_repo/tests/pypto_test/dsv4_hca_prepare_opp.py" \
    --destination "$hca_spec_output/private_opp")
export ASCEND_OPP_PATH
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
# CANN会把绝对编译目录编码进单个shape_info文件名；长结果路径会超过NAME_MAX。
# 仅缩短编译工作目录，报告/私有OPP/原始profiling仍保存在本轮结果目录。
hca_spec_compile=$(mktemp -d "$hca_spec_workspace/.cache/hca-perf-XXXXXX")
printf '%s\n' "$hca_spec_compile" > "$hca_spec_output/compile_workdir.txt"
cd "$hca_spec_compile"
exec python "$hca_spec_repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$hca_spec_workspace" \
    --vllm-root "$hca_spec_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$hca_spec_here/run_native.py" --output "$hca_spec_output" "$@"

#!/usr/bin/env bash
# 每个实验目录在提交前冻结；两侧源码/环境相同且不修改共享 SDK。
set -eo pipefail
experiment=${1:?experiment directory}
out=${2:?output directory}
case_name=${3:?fixed case name}
shift 3
repo=$(git -C "$experiment" rev-parse --show-toplevel)
workspace=$(dirname -- "$repo")
if [[ " $* " != *" --cpu "* ]]; then
    : "${TASK_DEVICE:?必须通过 task-submit 分配设备}"
fi
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=spmd_tnd
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
unset HC_PRE_TEST_OPERATOR_SOURCE
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" \
    --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$experiment/run_case.py" --output "$out" --case "$case_name" \
    --candidate "$experiment/operator" \
    --frozen "$repo/tests/pypto_test/csa_mpmd_tnd_20261021/sources/84751e34" "$@"

#!/usr/bin/env bash
# 完整单 SPMD 设备验证，只由 task-submit 分配单卡后运行。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 分配卡}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$here")"
root=/data/pyptouser/qinchuanyu/pto-eager
out="${1:?指定全新输出目录}"
case_name="${2:?指定已有静态档位，例如 h131072_b16}"
if [[ "$case_name" = /* ]]; then
    case_path="$case_name"
else
    case_path="$here/static/$case_name"
fi
graph_args=()
if [[ "${3:-}" == "--graph-replay" ]]; then graph_args=(--graph-replay); fi
source "$root/env-dsv4-0251rc1.sh"
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有验证目录：%s\n' "$out" >&2
    exit 1
fi
mkdir -p "$out/ascend"
export CSA_STATIC_PYPTO_ROOT="$root/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$root/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_BATCH_INVARIANT=0
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export HC_PRE_TEST_OPERATOR_SOURCE="$tests/results/hc_pre_layer_integration_20261012/baseline_ops"
export ASCEND_OPP_PATH="$(python "$tests/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
cd "$out"
python "$here/bootstrap_isolated.py" "$here/test_static_jit_device.py" \
    --case "$case_path" --output "$out" --device "$TASK_DEVICE" "${graph_args[@]}" \
    > "$out/run.log" 2>&1

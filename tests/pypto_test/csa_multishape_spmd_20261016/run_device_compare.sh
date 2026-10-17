#!/usr/bin/env bash
# 按真实输入形状准备变体。两侧均为 PTO，不运行 Native forward。
set -eo pipefail
: "${TASK_DEVICE:?必须通过 task-submit --device auto 运行}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$here")"
static="$tests/csa_static_spmd_feasibility_20261015"
root=/data/pyptouser/qinchuanyu/pto-eager
batch="${1:?指定 batch}"
history="${2:?指定 history}"
out="${3:?指定新的结果目录}"
source "$root/env-dsv4-0251rc1.sh"
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有结果：%s\n' "$out" >&2
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
export HC_PRE_TEST_OPERATOR_SOURCE="$static/reference_operator_source"
export ASCEND_OPP_PATH="$(python "$tests/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
cp "$here/device_compare.py" "$out/benchmark_driver.py"
cd "$out"
python "$static/bootstrap_isolated.py" "$here/device_compare.py" \
    --batch "$batch" --history "$history" --output "$out" --device "$TASK_DEVICE" \
    --warmup 5 --iters 20 --windows 3 "${@:4}" > "$out/run.log" 2>&1

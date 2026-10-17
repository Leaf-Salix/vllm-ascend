#!/usr/bin/env bash
# 单档 S6 调度补采，使用与历史泳道一致的冻结 baseline 算子。
# 通过 task-submit 执行；DEBUG 会改变调度耗时，结果不能用作性能比较。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配卡}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$here")"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
out="${1:?指定全新的输出目录}"
history=131072
batch=4
# task-submit 会在末尾追加 --device，只有显式数字参数才作为规格。
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then history="$2"; fi
if [[ "${3:-}" =~ ^[0-9]+$ ]]; then batch="$3"; fi
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有采集目录：%s\n' "$out" >&2
    exit 1
fi
mkdir -p "$out/ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_BATCH_INVARIANT=0
export HC_PRE_DIAGNOSTIC_LEVEL=1
# vLLM 会先于 Simpler 初始化 NPU 上下文；CANN 在那时固定设备日志级别。
# 仅设置 Python 的 simpler logger 太晚，必须在 Python 进程启动前设置。
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export HC_PRE_TEST_OPERATOR_SOURCE="$tests/results/hc_pre_layer_integration_20261012/baseline_ops"
export ASCEND_OPP_PATH="$(python "$tests/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
cd "$out"
python "$tests/hc_pre_layer_integration_20261012/diagnose_l1.py" \
    "$tests/hc_pre_layer_integration_20261012/csa_compiled_case.py" \
    --side pto --history "$history" --batch "$batch" --device "$TASK_DEVICE" \
    --super-kernel 0 --schedule-only --output "$out" > "$out/run.log" 2>&1
python "$here/build_schedule_manifest.py" --dfx "$out/dfx" \
    --final-replay-device-logs "$out/ascend" --output "$out/manifest" \
    > "$out/manifest_build.log" 2>&1
printf '单窗口派发补采完成：%s\n' "$out"

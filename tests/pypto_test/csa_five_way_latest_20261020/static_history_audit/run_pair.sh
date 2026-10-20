#!/usr/bin/env bash
# 一档在队列分配的同一卡上顺序重测五种口径；每个阶段独立进程，保留原始输出。
set -eo pipefail
: "${TASK_DEVICE:?必须经task-submit --device auto分配NPU}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
history=${1:?history}
batch=${2:?batch}
shift 2
requested_phase=pair
attempt=v1
if [[ $# -gt 0 && "$1" != --* ]]; then requested_phase=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then attempt=$1; shift; fi
# 队列会在命令末尾追加 --device N；只验证它，不将其误作阶段或自行改卡。
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            if [[ "$2" != "$TASK_DEVICE" ]]; then
                printf '队列设备不一致\n' >&2
                exit 2
            fi
            shift 2
            ;;
        *) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
    esac
done
case_dir="$here/results/h${history}_b${batch}"
mkdir -p "$case_dir"
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_SDMA_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-repro-20261019"
export CSA_SDMA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export CSA_STATIC_PYPTO_ROOT="$CSA_SDMA_PYPTO_ROOT" CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$CSA_SDMA_SIMPLER_ROOT"
export PYTHONPATH="$repo/tests/pypto_test/csa_critical_path_20261020/python_startup:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
bootstrap=(python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"
  --pypto-root "$CSA_SDMA_PYPTO_ROOT" --simpler-root "$CSA_SDMA_SIMPLER_ROOT"
  --ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build")

out="$case_dir/pair_${attempt}"
if [[ -e "$out" ]]; then printf '拒绝覆盖既有配对结果\n' >&2; exit 2; fi
export ASCEND_PROCESS_LOG_PATH="$case_dir/device_logs_${attempt}"
export ASCEND_CACHE_PATH="$case_dir/ascend_cache_${attempt}"
export PYPTO_PROG_BUILD_DIR="$out/no_compile_expected" PYPTO_CACHE_DIR="$out/cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
unset HC_PRE_TEST_OPERATOR_SOURCE
"${bootstrap[@]}" "$here/prebuilt_pair.py" --device "$TASK_DEVICE" --history "$history" \
    --batch "$batch" --output "$out" --l1-windows 6 > "$case_dir/pair_${attempt}.log" 2>&1

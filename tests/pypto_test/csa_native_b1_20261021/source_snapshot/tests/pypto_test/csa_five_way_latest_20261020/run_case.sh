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
requested_phase=all
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

run_phase() {
    phase=$1
    out="$case_dir/${phase}_${attempt}"
    if [[ -e "$out" ]]; then
        printf '拒绝覆盖已有结果：%s\n' "$out" >&2
        return 2
    fi
    printf '%s H%s/B%s %s START\n' "$(date -Is)" "$history" "$batch" "$phase"
    export ASCEND_PROCESS_LOG_PATH="$case_dir/${phase}_${attempt}_device_logs"
    export ASCEND_CACHE_PATH="$case_dir/${phase}_${attempt}_ascend_cache"
    export VLLM_CACHE_ROOT="$case_dir/${phase}_${attempt}_vllm_cache"
    export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
    mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
    unset HC_PRE_TEST_OPERATOR_SOURCE
    if [[ "$phase" == native || "$phase" == multitask_outer ]]; then
        mkdir -p "$out"
        export ASCEND_OPP_PATH="$(python "$repo/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
        mkdir -p "$ASCEND_OPP_PATH/static_kernel" "$workspace/.cache/a"
        compile_cwd=$(mktemp -d "$workspace/.cache/a/five_XXXXXX")
        printf '%s\n' "$compile_cwd" > "$out/compile_workdir.txt"
        if [[ "$phase" == native ]]; then
            (cd -- "$compile_cwd"; "${bootstrap[@]}" "$here/native_case.py" \
                --device "$TASK_DEVICE" --history "$history" --batch "$batch" --output "$out") \
                > "$case_dir/${phase}_${attempt}.log" 2>&1
        else
            export HC_PRE_TEST_OPERATOR_SOURCE="$repo/vllm_ascend/ops/pypto"
            (cd -- "$compile_cwd"; "${bootstrap[@]}" "$here/multitask_case.py" \
                --side pto --super-kernel 0 --device "$TASK_DEVICE" \
                --history "$history" --batch "$batch" --output "$out") \
                > "$case_dir/${phase}_${attempt}.log" 2>&1
        fi
    elif [[ "$phase" == l1 ]]; then
        (cd -- "$repo"; "${bootstrap[@]}" "$here/l1_case.py" \
            --device "$TASK_DEVICE" --history "$history" --batch "$batch" --output "$out" \
            --launches 3 --l1-windows 5 --wait-backoff --failure-snapshot \
            --spmd-source "$here/source_dynamic" \
            --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build") \
            > "$case_dir/${phase}_${attempt}.log" 2>&1
    elif [[ "$phase" == static ]]; then
        if [[ "$history" == 131072 && "$batch" == 28 ]]; then
            printf '128K/B28静态参考没有已验收实现，明确留空，不借邻档\n'
            return 0
        fi
        (cd -- "$repo"; "${bootstrap[@]}" "$here/static_case.py" \
            --device "$TASK_DEVICE" --history "$history" --batch "$batch" \
            --output "$out" --l1-windows 5) > "$case_dir/${phase}_${attempt}.log" 2>&1
    else
        printf '未知阶段：%s\n' "$phase" >&2
        return 2
    fi
    printf '%s H%s/B%s %s COMPLETE\n' "$(date -Is)" "$history" "$batch" "$phase"
}

if [[ "$requested_phase" == all ]]; then
    for step in native multitask_outer l1 static; do
        run_phase "$step"
    done
else
    run_phase "$requested_phase"
fi

#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?父任务使用自动分配的单卡，不嵌套提交}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_native_inplace_seven_20260929"
side="${1:?native, pto, native_incore, or swimlane}"
history="${2:?history}"
batch="${3:?batch}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source_repo="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$root/source.json")"
export PTO_CSA_VARIANT=pkg:dsv4_csa_native_inplace_seven_20260929
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_RING_HEAP_MB=256,128,256,32 PTO_CSA_RING_TASK_WINDOW=4096
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
out="$root/h${history}_b${batch}/$side"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
runner="$source_repo/tests/pypto_test/coefficients_seven_experiment"
if [[ "$side" == native || "$side" == pto || "$side" == native_incore ]]; then
    ASCEND_OPP_PATH="$(python "$runner/prepare_opp.py" --destination "$out/opp_env")"
    export ASCEND_OPP_PATH
    entry="$runner/compiled_case.py"
    extra=()
    runner_side="$side"
    if [[ "$side" == native ]]; then
        entry="$runner/native_case.py"
        extra=(--super-kernel 1)
    elif [[ "$side" == native_incore ]]; then
        entry="$runner/native_case.py"
        runner_side=native
        extra=(--super-kernel 0 --profile-only)
    fi
    cd "$out"
    exec python "$entry" --side "$runner_side" "${extra[@]}" --history "$history" --batch "$batch" \
        --output "$out" --device "$TASK_DEVICE" > "$out/run.log" 2>&1
elif [[ "$side" == swimlane ]]; then
    cd "$out"
    exec python "$runner/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
        --layer-index 4 --variant "$PTO_CSA_VARIANT" --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 0 --swimlane --swimlane-graph --swimlane-windows 4 \
        > "$out/run.log" 2>&1
else
    echo "Unsupported phase: $side" >&2
    exit 2
fi

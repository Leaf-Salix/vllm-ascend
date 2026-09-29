#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit the five missing cases through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_single_root_seven_20260929"
side="${1:?baseline or candidate}"
phase="${2:?timing or swimlane}"
history="${3:?history length}"
batch="${4:?batch size}"
[[ "$side" == baseline || "$side" == candidate ]]
[[ "$phase" == timing || "$phase" == swimlane ]]
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source_repo="$workspace/.cache/csa-score-single-root-d93bba14-$side"
export PTO_CSA_VARIANT=pkg:dsv4_csa_score_single_root_d93bba14
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_RING_HEAP_MB=256,128,256,32 PTO_CSA_RING_TASK_WINDOW=4096
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
out="$root/h${history}_b${batch}/$phase/$side"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
if [[ "$phase" == timing ]]; then
    ASCEND_OPP_PATH="$(python "$source_repo/tests/pypto_test/coefficients_seven_experiment/prepare_opp.py" \
        --destination "$out/opp_env")"
    export ASCEND_OPP_PATH
    cd "$out"
    exec python "$source_repo/tests/pypto_test/coefficients_seven_experiment/compiled_case.py" \
        --side pto --history "$history" --batch "$batch" --output "$out" \
        --device "$TASK_DEVICE" > "$out/run.log" 2>&1
else
    cd "$out"
    exec python "$source_repo/tests/pypto_test/coefficients_seven_experiment/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
        --layer-index 4 --variant "$PTO_CSA_VARIANT" --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 0 --swimlane --swimlane-graph --swimlane-windows 4 \
        > "$out/run.log" 2>&1
fi

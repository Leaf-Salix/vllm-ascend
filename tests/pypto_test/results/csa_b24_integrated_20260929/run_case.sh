#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_b24_integrated_20260929"
pair="$repo/tests/pypto_test/results/csa_compiled_pair_20260929"
label="${1:?baseline or candidate}"
shift
source_repo="$workspace/.cache/csa-b24-integrated-c93ec723-$label"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=pkg:dsv4_csa_b24_integrated_c93ec723
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
side="${1:?native or pto}"
history="${2:?history length}"
batch="${3:?batch size}"
out="$root/final/h${history}_b${batch}/${side}_$label"
extra=()
if [[ "$side" == pto ]]; then extra=(--save-state); fi
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
ASCEND_OPP_PATH="$(python "$pair/prepare_opp.py" --destination "$out/opp_env")"
export ASCEND_OPP_PATH
cd "$out"
exec python "$root/compiled_case.py" --side "$side" --history "$history" --batch "$batch" \
    --output "$out" --device "$TASK_DEVICE" "${extra[@]}" > "$out/run.log" 2>&1

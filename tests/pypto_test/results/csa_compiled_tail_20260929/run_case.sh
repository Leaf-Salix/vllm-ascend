#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_compiled_tail_20260929"
pair="$repo/tests/pypto_test/results/csa_compiled_pair_20260929"
source_repo="$workspace/.cache/csa-small-long-s6-d8627207-baseline"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=pkg:dsv4_csa_small_long_s6_d8627207
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
side="${1:?native or pto}"
history="${2:?history length}"
batch="${3:?batch size}"
out="$root/h${history}_b${batch}/$side"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
ASCEND_OPP_PATH="$(python "$pair/prepare_opp.py" --destination "$out/opp_env")"
export ASCEND_OPP_PATH
cd "$out"
exec python "$root/diagnose.py" --side "$side" --history "$history" --batch "$batch" \
    --output "$out" --device "$TASK_DEVICE" > "$out/run.log" 2>&1

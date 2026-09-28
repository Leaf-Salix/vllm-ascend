#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_native_compiled_layer_20260929"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$repo/tests/pypto_test:${PYTHONPATH:-}"
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2 DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
out="$root/h131072_b16"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
cd "$out"
exec python "$root/native_layer.py" --history 131072 --batch 16 --output "$out" --device "$TASK_DEVICE" \
    > "$out/run.log" 2>&1

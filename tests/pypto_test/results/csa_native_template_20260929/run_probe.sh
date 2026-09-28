#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_native_template_20260929"
source "$workspace/env-dsv4-0251rc1.sh"
source "$root/env.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$repo:$repo/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_ENABLE_NZ=2 OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_OP_EXPANSION_MODE=AIV
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$root"
exec python "$root/probe.py" > "$root/probe.log" 2>&1

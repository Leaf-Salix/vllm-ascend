#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_ob_hc_scalar_fused_20260929"
source "$workspace/env-dsv4-0251rc1.sh"
export PTO_CSA_VARIANT=pkg:dsv4_csa_ob_hc_scalar_fused_9a868d26
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export OMP_NUM_THREADS=4 HCCL_DETERMINISTIC=true
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
for side in baseline candidate; do
    source_repo="$workspace/.cache/csa-ob-hc-scalar-fused-9a868d26-$side"
    out="$root/shared_hc/$side"
    test ! -e "$out/report.json"
    mkdir -p "$out/ascend" "$out/ascend_cache"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
    cd "$out"
    python "$root/shared_hc_case.py" --source "$source_repo" --output "$out" \
        --device "$TASK_DEVICE" > "$out/run.log" 2>&1
done

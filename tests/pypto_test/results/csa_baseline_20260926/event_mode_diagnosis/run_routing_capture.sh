#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 16 卡}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env-dsv4-0251rc1.sh
root="$PWD/tests/pypto_test/results/csa_baseline_20260926/event_mode_diagnosis/routing"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export HCCL_DETERMINISTIC=false
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py moe-routing \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h131072_bank \
        --output "$root/$backend" --backend "$backend" --batch 40 --rank-batches 4 \
        --max-num-batched-tokens 256 --event-work-mode 1 \
        --decode-tokens 128 --weight-nz-mode 2 --warmup-rounds 1 --warmup-tokens 96 \
        --warmup-steps 8 --compare-samples 3 \
        --graph-mode full_decode_only --capture-sizes 24 48 96 144 192 240
done

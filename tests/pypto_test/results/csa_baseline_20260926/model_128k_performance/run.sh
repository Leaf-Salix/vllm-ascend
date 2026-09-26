#!/usr/bin/env bash
# 128K 泛化：容量固定 40，每侧加载一次，实际 batch 扫描六档；EPLB 关闭。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 16 卡}"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
result_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/model_128k_performance/capacity40"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export HCCL_DETERMINISTIC=false
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py performance \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h131072_bank \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 40 --sweep-batches 4 8 16 24 32 40 --decode-tokens 192 --weight-nz-mode 2 \
        --warmup-rounds 1 --warmup-tokens 96 --warmup-steps 8 \
        --profile-start-step 8 --profile-steps 3 \
        --graph-mode full_decode_only --capture-sizes 24 48 96 144 192 240
done

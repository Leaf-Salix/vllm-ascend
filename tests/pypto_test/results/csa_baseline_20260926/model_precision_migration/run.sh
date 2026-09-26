#!/usr/bin/env bash
# 精度版迁移后的正式模型 token/DSpark 看护，两侧 EPLB 关闭、mode=2。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 16 卡}"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
result_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/model_precision_migration"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=precision
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=true
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py decode \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 16 --decode-tokens 96 --weight-nz-mode 2 --deterministic \
        --graph-mode full_decode_only --capture-sizes 96
done
python tests/pypto_test/offline_pd/compare.py \
    --native "$result_root/native" --pto "$result_root/pto" \
    --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
    --batch 16 --decode-tokens 96 --ranks 16 --output "$result_root/comparison.json"

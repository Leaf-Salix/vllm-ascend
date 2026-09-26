#!/usr/bin/env bash
# 一次加载分别采无 profiler 稳态步和独立 Level0 层区间，两侧使用同一 mode。
set -eo pipefail
mode="${1:?指定 NZ mode 1 或 2}"
[[ "$mode" == 1 || "$mode" == 2 ]]
result_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mode$mode"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export HCCL_DETERMINISTIC=false
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py performance \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 16 --decode-tokens 192 --weight-nz-mode "$mode" \
        --warmup-rounds 1 --warmup-tokens 96 --warmup-steps 8 \
        --profile-start-step 8 --profile-steps 3 \
        --graph-mode full_decode_only --capture-sizes 96
done
